import os
import torch
import matplotlib.pyplot as plt
from itertools import chain
from transformers import AdamW
from datasets import load_dataset
import wandb
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

model_path = "models/Qwen2.5-0.5B-Instruct"
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

############
# 数据处理略过，参考实践篇，seq_len设置为256
############

# 训练参数配置
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-8,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    save_total_limit=3,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=20,
    report_to="wandb",
)

wandb.init(project="qwen-0.5B-pt", config=training_args)

optimizer = AdamW(model.parameters(), lr=1e-8, weight_decay=0.01)

trainer = Trainer(model=model, args=training_args, data_collator=collator, train_dataset=train_dataset, optimizers=(optimizer, None))

torch.cuda.empty_cache()
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(output_path)
