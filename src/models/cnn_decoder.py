import torch
import torch.nn as nn
import numpy as np
from firelab.utils import cudable


class CNNDecoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, latent_size,
                 kernel_size=3, dilations=(1,2,4)):
        super(CNNDecoder, self).__init__()

        self.hid_size = hid_size
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        first_layer = nn.Conv1d(emb_size+latent_size, hid_size, kernel_size, dilation=dilations[0])
        other_layers = [nn.Conv1d(hid_size+latent_size, hid_size, kernel_size, dilation=d) for d in dilations[1:]]

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
            out = self.layers[i](out.transpose(1,2)).transpose(1,2)
            out = self.activations[i](out)

        out = self.embs_to_logits(out)
        out = self.out_activation(out)

        return out

    def inference(self, z, vocab, max_len=100):
        batch_size = z.size(0)
        BOS, EOS = vocab.stoi['<bos>'], vocab.stoi['<eos>']
        active_seqs = cudable(torch.tensor([[BOS] for _ in range(batch_size)]).long())
        active_seqs_idx = np.arange(batch_size)
        finished = [None for _ in range(batch_size)]
        n_finished = 0

        for _ in range(max_len):
            next_tokens = self.forward(z, active_seqs).max(dim=-1)[1][:,-1] # TODO: use beam search
            active_seqs = torch.cat((active_seqs, next_tokens.unsqueeze(1)), dim=-1)
            finished_mask = (next_tokens == EOS).cpu().numpy().astype(bool)
            finished_seqs_idx = active_seqs_idx[finished_mask]
            active_seqs_idx = active_seqs_idx[finished_mask == 0]
            n_finished += finished_seqs_idx.size

            if finished_seqs_idx.size != 0:
                # TODO(universome)
                # finished[finished_seqs_idx] = active_seqs.masked_select(next_tokens == EOS).cpu().numpy()
                for i, seq in zip(finished_seqs_idx, active_seqs[next_tokens == EOS]):
                    finished[i] = seq.cpu().numpy().tolist()

                active_seqs = active_seqs[next_tokens != EOS]
                z = z[next_tokens != EOS]

            if n_finished == batch_size: break

        # Well, some sentences were finished at the time
        # Let's just fill them in
        if n_finished != batch_size:
            # TODO(universome): finished[active_seqs_idx] = active_seqs
            for i, seq in zip(active_seqs_idx, active_seqs):
                finished[i] = seq.cpu().numpy().tolist()

        return finished


def shift_sequence(seq, n):
    """Prepends each sequence in a batch with n zero vectors"""
    batch_size, vec_size = seq.size(0), seq.size(-1)
    shifts = cudable(torch.zeros(batch_size, n, vec_size))

    return torch.cat((shifts, seq), dim=1)


def concatenate_z(seq, z):
    """Concatenates latent vector z to each word embedding of the sequence"""
    seq_len = seq.size(1)
    zs = z.unsqueeze(1).repeat(1, seq_len, 1)

    return torch.cat((seq, zs), dim=2)
