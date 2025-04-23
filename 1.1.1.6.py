from collections import defaultdict

# 示例语料
sentences = [
    "我",
    "喜欢",
    "吃",
    "苹果",
    "他",
    "不",
    "喜欢",
    "吃",
    "苹果派",
    "I like to eat apples",
    "She has a cute cat",
    "you are very cute",
    "give you a hug",
]

# 构建初始词汇表，包含所有256个字节
initial_vocab = [bytes([byte]) for byte in range(256)]
vocab = initial_vocab.copy()

# 存储合并规则
merge_rules = []

print("初始词汇表:", initial_vocab)


# 构建频率统计
def build_stats(sentences):
    stats = defaultdict(int)
    for sentence in sentences:
        # 将句子编码为UTF-8字节
        symbols = sentence.encode("utf-8").split()
        for symbol in symbols:
            stats[symbol] += 1
    return stats


stats = build_stats(sentences)

# 初始化分割方案：每个词由其字节组成的列表表示（字节对象）
splits = {word: [bytes([byte]) for byte in word] for word in stats.keys()}


# 计算字节对的频率
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in stats.items():
        split = splits[word]
        if len(split) < 2:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


# 合并指定的字节对，并更新分割方案
def merge_pair(pair, splits):
    merged_byte = b"".join(pair)
    for word in stats:
        split = splits[word]
        if len(split) < 2:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i : i + 2] == list(pair):
                # 替换匹配的字节对为合并后的字节
                split = split[:i] + [merged_byte] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


# BPE 主过程
def byte_level_bpe(sentences, initial_vocab, target_vocab_size=50):
    vocab = initial_vocab.copy()
    merge_rules = []
    stats = build_stats(sentences)
    splits = {word: [bytes([byte]) for byte in word] for word in stats.keys()}

    while len(vocab) < target_vocab_size:
        pair_freqs = compute_pair_freqs(splits)
        if not pair_freqs:
            print("没有更多的字节对可以合并。")
            break
        # 找到频率最高的字节对
        best_pair, max_freq = max(pair_freqs.items(), key=lambda x: x[1])
        if max_freq < 1:
            print("最高频率的字节对频率小于1，停止合并。")
            break
        # 合并字节对
        splits = merge_pair(best_pair, splits)
        # 创建新的子词单元
        new_unigram = b"".join(best_pair)
        # 添加合并规则
        merge_rules.append((best_pair, new_unigram))
        # 更新词汇表
        vocab.append(new_unigram)
        print(f"合并字节对: {best_pair} -> {new_unigram}")
        print(f"当前词汇表大小: {len(vocab)}")

    return vocab, merge_rules


# 运行 BBPE
vocab_size = 260
vocab, merge_rules = byte_level_bpe(sentences, initial_vocab, target_vocab_size=vocab_size)

print("\n最终词汇表:", vocab)
print("\n合并规则:")
for i, rule in enumerate(merge_rules, 1):
    print(f"{i}: {rule[0]} -> {rule[1]}")
