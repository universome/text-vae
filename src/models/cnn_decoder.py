import torch
import torch.nn as nn
import numpy as np
from firelab.utils import cudable

from .utils import concatenate_z

class CNNDecoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, latent_size,
                 kernel_size=3, dilations=(1,2,4), dropout_rate=0.1):
        super(CNNDecoder, self).__init__()

        self.hid_size = hid_size
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        first_layer = nn.Conv1d(emb_size+latent_size, hid_size, kernel_size, dilation=dilations[0])
        other_layers = [nn.Conv1d(hid_size+latent_size, hid_size, kernel_size, dilation=d) for d in dilations[1:]]

        self.dropouts = [nn.Dropout(dropout_rate) for _ in dilations]
        self.layers = nn.ModuleList([first_layer] + other_layers)
        self.activations = [nn.SELU() for _ in self.dilations]

        self.embs_to_logits = nn.Linear(hid_size + latent_size, vocab_size)
        self.embs_to_logits.weight = self.embeddings.weight # Sharing weights
        self.out_activation = nn.SELU()

    def forward(self, z, sentence):
        # We do not apply masks. Instead, we shift input sequence with pad tokens
        shifts = [(self.kernel_size - 1) * d for d in self.dilations]
        out = self.embeddings(sentence)

        for i in range(len(self.layers)):
            out = shift_sequence(out, shifts[i])
            out = concatenate_z(out, z)
            out = self.dropouts[i](out)
            out = self.layers[i](out.transpose(1,2)).transpose(1,2)
            out = self.activations[i](out)

        out = self.embs_to_logits(out)
        out = self.out_activation(out)

        return out


def shift_sequence(seq, n):
    """Prepends each sequence in a batch with n zero vectors"""
    batch_size, vec_size = seq.size(0), seq.size(-1)
    shifts = cudable(torch.zeros(batch_size, n, vec_size))

    return torch.cat((shifts, seq), dim=1)
