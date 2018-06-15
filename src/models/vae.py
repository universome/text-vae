import torch
import torch.nn as nn
import numpy as np

from .lstm_encoder import LSTMEncoder
from .cnn_decoder import CNNDecoder

use_cuda = torch.cuda.is_available()

class VAE(nn.Module):
    def __init__(self, emb_size, hid_size, vocab_size, latent_size):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder = LSTMEncoder(emb_size, hid_size, vocab_size, latent_size)
        self.decoder = CNNDecoder(emb_size, hid_size, vocab_size, latent_size)

        if torch.cuda.device_count() > 1:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

    def forward(self, inputs, targets):
        # inputs, targets = batch.text[:, :-1], batch.text[:, 1:]
        encodings = self.encoder(inputs)
        means, stds = encodings[:, :32], encodings[:, 32:]
        latents = sample(means, stds)
        out = self.decoder(latents, targets)

        return latents, out


def sample(means, stds):
    noise = torch.from_numpy(np.random.normal(size=stds.size())).float()
    if use_cuda: noise = noise.cuda()
    latents = means + stds * noise

    return latents
