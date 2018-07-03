import torch.nn as nn

from .layers import Dropword


class RNNDecoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, latent_size, dropword=0):
        super(RNNDecoder, self).__init__()

        self.hid_size = hid_size
        self.dropword = Dropword(dropword)
        self.z_to_state = nn.Sequential(
            nn.Linear(latent_size, hid_size),
            nn.SELU()
        )
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)
        self.embs_to_logits = nn.Linear(hid_size, vocab_size)
        # self.embs_to_logits.weight = self.embeddings.weight # Sharing weights

    def forward(self, z, sentences, dropword_p=0):
        # First, let's compute decoder initial state based on latent vector
        state = self.z_to_state(z).unsqueeze(0).contiguous()
        embeds = self.embeddings(sentences)
        embeds = self.dropword(embeds, dropword_p)
        hid_states, _ = self.gru(embeds, state)
        logits = self.embs_to_logits(hid_states)

        return logits
