"""
This module contains common helper functions
"""

SPECIAL_TOKENS = {'<bos>', '<eos>', '<pad>'}


def itos_many(seqs, vocab):
    """
    Converts sequences of token ids to normal strings
    """
    sents = [[vocab.itos[i] for i in seq] for seq in seqs]
    sents = [[t for t in s if not t in SPECIAL_TOKENS] for s in sents]
    sents = [' '.join(s).replace('@@ ', '') for s in sents]

    return sents
