import torch
import torch.nn as nn
import numpy as np
from firelab.utils import cudable


class VAE(nn.Module):
    def __init__(self, encoder:nn.Module, decoder:nn.Module, latent_size:int):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder = nn.DataParallel(encoder)
        self.decoder = nn.DataParallel(decoder)

    def forward(self, inputs:torch.Tensor, targets):
        # inputs, targets = batch.text[:, :-1], batch.text[:, 1:]
        encodings = self.encoder(inputs)
        means, stds = encodings[:, :32], encodings[:, 32:]
        latents = sample(means, stds)
        out = self.decoder(latents, targets)

        return latents, out

    def inference(self, inputs:torch.Tensor, vocab):
        """Performs inference on raw sentences"""
        encodings = self.encoder(inputs)
        means, stds = encodings[:, :32], encodings[:, 32:]
        latents = sample(means, stds)
        sentences = self.decoder.module.inference(latents, vocab)

        return sentences


def sample(means, stds):
    noise = cudable(torch.from_numpy(np.random.normal(size=stds.size())).float())
    latents = means + stds * noise

    return latents
