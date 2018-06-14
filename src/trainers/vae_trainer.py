import os

import torch
from torch.optim import Adam
import numpy as np
import pandas as pd
from firelab import BaseTrainer

from src.models.vae import sample

use_cuda = torch.cuda.is_available()


class VAETrainer(BaseTrainer):
    def __init__(self, vae, optimizer, config, on_iter_done_callback=None):
        self.vae = vae
        self.optimizer = optimizer
        self.config = config
        self.kl_beta_coef = self.config['hp'].get('kl_beta_coef', 0.5)

        assert list(self.vae.parameters()) != []

        self.rec_loss_history = []
        self.kl_loss_history = []

        self.rec_loss_file = open(os.path.join(config['firelab']['logs_dir'], 'rec_loss.csv'), 'w')
        self.kl_loss_file = open(os.path.join(config['firelab']['logs_dir'], 'kl_loss.csv'), 'w')

        self.num_iters_done = 0
        self.max_num_epochs = config.get('max_num_epochs', 10)
        self.plot_every = config.get('plot_every', 50)
        self.val_bleu_every = config.get('val_bleu_every', 100)

        # TODO: Ohh. You loved javascript in 2007, I guess?
        self.on_iter_done_callback = on_iter_done_callback

    def run_training(self, training_data, val_data):
        should_continue = True

        while self.num_epochs_done < self.max_num_epochs and should_continue:
            try:
                for batch in tqdm(training_data, leave=False):
                    self.train_on_batch(batch)
                    if self.num_iters_done % val_bleu_every == 0: self.validate_bleu(val_data)
                    if self.num_iters_done % plot_every == 0: self.plot_scores()
                    self.num_iters_done += 1
            except KeyboardInterrupt:
                should_continue = False
                break

            self.num_epochs_done += 1

        self.rec_loss_file.close()
        self.kl_loss_file.close()

    def train_on_batch(self, batch):
        inputs, targets = batch.text[:, :-1], batch.text[:, 1:]
        encodings = self.vae.encoder(inputs)
        means, stds = encodings[:, :self.vae.latent_size], encodings[:, self.vae.latent_size:]
        latents = sample(means, stds)
        reconstructions = self.vae.decoder(latents, inputs)

        rec_loss = rec_criterion(reconstructions.view(-1, vocab_size), targets.contiguous().view(-1))
        kl_loss = kl_criterion(means, stds)
        loss = rec_loss + self.kl_beta_coef * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rec_loss_history.append(rec_loss.item())
        self.kl_loss_history.append(kl_loss.item())
        self.log_losses()

    def log_losses(self):
        self.rec_loss_file.write(str(self.rec_loss_history[-1]) + '\n')
        self.kl_loss_file.write(str(self.kl_loss_history[-1]) + '\n')

        if not self.on_iter_done_callback is None:
            self.on_iter_done_callback()
