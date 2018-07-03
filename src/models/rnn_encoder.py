import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, latent_size):
        super(RNNEncoder, self).__init__()

        self.hid_size = hid_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)
        self.hid_to_z = nn.Sequential(
            nn.Linear(hid_size, 256),
            nn.SELU(),
            nn.Linear(256, latent_size * 2)
        )

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        _, last_hidden_state = self.gru(embeds)
        out = self.hid_to_z(last_hidden_state.squeeze(0))

        return out
