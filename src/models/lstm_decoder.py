import torch.nn as nn

from .layers import Dropword


class LSTMDecoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, latent_size, dropword=0):
        super(LSTMDecoder, self).__init__()

        self.hid_size = hid_size
        self.dropword = Dropword(dropword)
        self.z_to_states = nn.Sequential(
            nn.Linear(latent_size, hid_size * 2),
            nn.SELU()
        )
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True)
        self.embs_to_logits = nn.Linear(hid_size, vocab_size)
        self.embs_to_logits.weight = self.embeddings.weight # Sharing weights

    def forward(self, z, sentences):
        # First, let's compute decoder initial state based on latent vector
        initial_states = self.z_to_states(z)
        states = initial_states[:, :self.hid_size], initial_states[:, self.hid_size:]
        states = (states[0].unsqueeze(0).contiguous(), states[1].unsqueeze(0).contiguous())

        embeds = self.embeddings(sentences)
        embeds = self.dropword(embeds)
        hidden_states, _ = self.lstm(embeds, states)
        logits = self.embs_to_logits(hidden_states)

        return logits
