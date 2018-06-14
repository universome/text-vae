import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, latent_size):
        super(LSTMEncoder, self).__init__()

        self.hid_size = hid_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True)
        self.hid_to_z = nn.Sequential(
            nn.Linear(hid_size, 256),
            nn.SELU(),
            nn.Linear(256, latent_size * 2)
        )

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        _, (_, out) = self.lstm(embeds)
        dist = self.hid_to_z(out.squeeze(0))

        return dist
