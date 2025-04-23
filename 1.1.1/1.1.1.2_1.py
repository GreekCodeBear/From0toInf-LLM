def forward_max_matching(sentence, dictionary, max_len=6):
    result = []
    i = 0
    while i < len(sentence):
        matched = False
        for j in range(max_len, 0, -1):
            if i + j > len(sentence):
                continue
            word = sentence[i : i + j]
            if word in dictionary:
                result.append(word)
                i += j
                matched = True
                break
        if not matched:
            result.append(sentence[i])
            i += 1
    return result


# 示例词典
dictionary = {"商务处", "女干事", "商务", "处女", "干事"}

# 示例句子
sentence = "商务处女干事"

# 分词
print("/".join(forward_max_matching(sentence, dictionary)))
# 输出: 商务处/女干事
