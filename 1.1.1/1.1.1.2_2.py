def backward_max_matching(sentence, dictionary, max_len=6):
    result = []
    i = len(sentence)
    while i > 0:
        matched = False
        for j in range(max_len, 0, -1):
            if i - j < 0:
                continue
            word = sentence[i - j : i]
            if word in dictionary:
                result.insert(0, word)
                i -= j
                matched = True
                break
        if not matched:
            result.insert(0, sentence[i - 1])
            i -= 1
    return result


# 示例词典
dictionary = {"商务处", "女干事", "商务", "处女", "干事"}

# 示例句子
sentence = "商务处女干事"

# 分词
print("/".join(backward_max_matching(sentence, dictionary)))
# 输出: 商务处/女干事
