from sentence_transformers import SentenceTransformer, util

# 1. 加载预训练好的SBERT模型
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. 准备句子
sentences = ["I love machine learning!", "Machine learning is my passion.", "The sky is clear today."]

# 3. 编码成向量
embeddings = model.encode(sentences)

# 4. 计算相似度
cos_sim = util.cos_sim(embeddings[0], embeddings[1])
print("Similarity between 0 and 1:", cos_sim.item())

cos_sim_01_2 = util.cos_sim(embeddings[0], embeddings[2])
print("Similarity between 0 and 2:", cos_sim_01_2.item())


from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

# 假设有STS数据： (sentence1, sentence2, similarity_score)
train_examples = [InputExample(texts=["I like apples", "Apples are my favorite fruit"], label=0.9), InputExample(texts=["She loves cats", "She hates cats"], label=0.2), ...]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 选用CosineSimilarityLoss
train_loss = losses.CosineSimilarityLoss(model)

# 训练
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, output_path="./fine_tuned_sbert")
