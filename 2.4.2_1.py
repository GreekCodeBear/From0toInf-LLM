import torch
from transformers import GPT2Tokenizer, GPT2Model

# 加载 GPT-2 模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# 输入文本
text = "This is an example sentence."

# 对输入文本进行编码
input_ids = tokenizer.encode(text, return_tensors="pt")

# 获取模型的输出
with torch.no_grad():
    outputs = model(input_ids)

# 提取最后一层隐藏层向量
last_hidden_state = outputs.last_hidden_state

# 获取 EOS 标记的 ID
eos_token_id = tokenizer.eos_token_id

# 寻找 EOS 标记在输入序列中的位置
eos_position = (input_ids == eos_token_id).nonzero(as_tuple=True)[1].item()

# 获取 EOS 对应的最后一层隐藏层向量
eos_hidden_vector = last_hidden_state[0, eos_position, :]

print("EOS hidden vector:", eos_hidden_vector)
