while len(vocab) < target_vocab_size:
        # 统计新词表导致的bigram频率
    bigram_score = get_pmi(word_freq)
        # 找到频率最大的bigram
    best_bigram = argmax(bigram_score)
        # 新词为频率最大的bigram的连接
    new_unigram = ''.join(best_bigram)
        # 对词频表中每个词应用best bigram的合并
    word_freq = merge_bigram(best_bigram, new_unigram, word_freq)
        #添加合并规则、添加新词
    merge_rule.append( {best_bigram ->new_unigram})
    vocab.append(new_unigram)