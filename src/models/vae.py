import torch
import torch.nn as nn
import numpy as np
from firelab.utils import cudable

from .utils import inference


class VAE(nn.Module):
    def __init__(self, encoder:nn.Module, decoder:nn.Module, latent_size:int):
        super(VAE, self).__init__()

        self.latent_size = latent_size
        self.encoder = nn.DataParallel(encoder)
        self.decoder = nn.DataParallel(decoder)

    def forward(self, inputs:torch.Tensor, targets):
        # inputs, targets = batch.text[:, :-1], batch.text[:, 1:]
        encodings = self.encoder(inputs)
        means, stds = encodings[:, :self.latent_size], encodings[:, self.latent_size:]
        latents = sample(means, stds)
        out = self.decoder(latents, targets)

        return latents, out

    def inference(self, inputs:torch.Tensor, vocab):
        """Performs inference on raw sentences"""
        encodings = self.encoder(inputs)
        means, log_stds = encodings[:, :self.latent_size], encodings[:, self.latent_size:]
        latents = sample(means, log_stds.exp())
        sentences = inference(self.decoder.module, latents, vocab)

        return sentences


def sample(means, stds):
    noise = cudable(torch.from_numpy(np.random.normal(size=stds.size())).float())
    latents = means + stds * noise

    return latents
