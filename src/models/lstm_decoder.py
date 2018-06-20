import torch.nn as nn

from .utils import concatenate_z


class LSTMDecoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, latent_size):
        super(LSTMDecoder, self).__init__()

        self.hid_size = hid_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size+latent_size, hid_size, batch_first=True)
        self.embs_to_logits = nn.Linear(hid_size + latent_size, vocab_size)
        self.embs_to_logits.weight = self.embeddings.weight # Sharing weights

    def forward(self, z, sentence):
        embeds = self.embeddings(sentence)
        embeds = concatenate_z(embeds, z)
        hidden_states, _ = self.lstm(embeds)
        logits = self.embs_to_logits(hidden_states)

        return logits
