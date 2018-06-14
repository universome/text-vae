def stoi_many(seqs, vocab):
    sents = [[vocab.itos[i] for i in s] for s in seqs]
    sents = [' '.join(s).replace('@@ ', '') for s in sents]

    return sents
