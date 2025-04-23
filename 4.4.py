from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import pipeline, TrainingArguments, Trainer

# 分词器
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")


# 函数内将instruction和response拆开分词的原因是：
# 为了便于mask掉不需要计算损失的labels, 即代码labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":
    # 加载数据集
    dataset = load_from_disk("./PEFT/data/alpaca_data_zh")

    # 处理数据
    tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)
    # print(tokenizer.decode(tokenized_ds[1]["input_ids"]))
    # print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"]))))

    # 创建模型
    model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh", low_cpu_mem_usage=True)
    # 基于bitfit只训练带有bias的参数
    for name, param in model.named_parameters():
        if "bias" not in name:
            param.requires_grad = False

    # 训练参数
    args = TrainingArguments(output_dir="./chatbot", per_device_train_batch_size=1, gradient_accumulation_steps=8, logging_steps=10, num_train_epochs=1)

    # trainer
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))

    # 训练模型
    trainer.train()

    # 模型推理
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    ipt = "Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: "
    output = pipe(ipt, max_length=256, do_sample=True)
    print(output)
