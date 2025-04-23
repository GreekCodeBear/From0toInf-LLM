# 安装: pip install transformers
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# 1. 加载预训练BERT (base)
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 构造数据示例
texts = ["I love this product!", "This is really bad..."]
labels = [1, 0]

encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
# encodings: {'input_ids': [[101, ... , 102], [...]], 'attention_mask': [...], 'token_type_ids':[...]}

import torch


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


dataset = SimpleDataset(encodings, labels)

# 3. Trainer 训练
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=2, logging_steps=10)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=...
)
trainer.train()

# 4. 推理
test_text = "I hate this!"
test_encoding = tokenizer(test_text, return_tensors="pt")
output = model(**test_encoding)
logits = output.logits
pred_label = logits.argmax(dim=1).item()
print("Prediction:", pred_label)
