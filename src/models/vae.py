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

        self.encodings_to_log_stds = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2),
            nn.SELU(),
            nn.Linear(latent_size // 2, latent_size)
        )

    def forward(self, seqs:torch.Tensor, noiseness=1, dropword_p=0):
        encodings = self.encoder(seqs)
        # TODO: well, actually, it is not cool to predict std from mean, but can we do?
        means, log_stds = encodings, self.encodings_to_log_stds(encodings)
        latents = sample(means, noiseness * log_stds.exp())
        out = self.decoder(latents, seqs[:, :-1], dropword_p=dropword_p)

        return (means, log_stds), out

    def inference(self, inputs:torch.Tensor, vocab, noiseness=1):
        """Performs inference on raw sentences"""
        encodings = self.encoder(inputs)
        means, log_stds = encodings, self.encodings_to_log_stds(encodings)
        latents = sample(means, noiseness * log_stds.exp())
        predictions = inference(self.decoder, latents, vocab)

        return predictions


def sample(means, stds):
    noise = cudable(torch.from_numpy(np.random.normal(size=stds.size())).float())
    latents = means + stds * noise

    return latents
