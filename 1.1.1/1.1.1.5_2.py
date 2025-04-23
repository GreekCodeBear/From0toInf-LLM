def bpe(self, token):
    if token in self.cache:
        return self.cache[token]

    word = tuple(token)  # (t, h, i, s, ...)
    pairs = get_pairs(word)  # {(t,h), (h,i), (i,s)}

    if not pairs:
        return token

    while True:
        # 1. 在所有当前 pairs 里，找到在 self.bpe_ranks 中 rank 最小（=优先级最高）的那对 bigram
        bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))

        # 如果这个 bigram (对) 不在 bpe_ranks 里，就说明后续都无法合并了，break
        if bigram not in self.bpe_ranks:
            break

        # 2. 在 word 里把这个 bigram 合并成一个新符号
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                # 找不到 first, 直接把后续都 append 到 new_word
                new_word.extend(word[i:])
                break
            else:
                new_word.extend(word[i:j])
                i = j

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                # 发现相邻 (first, second), 合并为 first+second
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        # word 替换为合并后的 new_word
        word = tuple(new_word)

        # 若只剩1个符号，没法再合并了
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # 最终, word 可能会变成多个 subword
    word = " ".join(word)
    self.cache[token] = word
    return word
