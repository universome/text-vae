import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseTrainer
from firelab.utils import cudable, HPLinearScheme, compute_param_by_scheme

from src.models import VAE, LSTMEncoder, LSTMDecoder, CNNDecoder
from src.models.vae import sample
from src.losses import KLLoss, compute_bleu_for_sents
from src.utils.common import itos_many


class VAETrainer(BaseTrainer):
    def __init__(self, config):
        super(VAETrainer, self).__init__(config)

        self.rec_loss_history = []
        self.kl_loss_history = []
        self.val_rec_loss_history = []
        self.val_kl_loss_history = []
        self.val_bleu_scores = []
        self.predictions = []

    def init_dataloaders(self):
        batch_size = self.config.get('batch_size', 8)
        project_path = self.config['firelab']['project_path']
        data_path = os.path.join(project_path, 'data/yelp-reviews.tok.bpe.random100k')

        with open(data_path) as f: lines = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)

        examples = [Example.fromlist([s], [('text', text)]) for s in lines]
        dataset = Dataset(examples, [('text', text)])
        # TODO: torchtext is insane. We pass split ratio for [train, val, test]
        # and it returns splits for [train, test, val]
        splits = dataset.split(split_ratio=[0.999, 0.0009, 0.0001])
        self.train_ds, self.test_ds, self.val_ds = splits
        text.build_vocab(self.train_ds)

        self.vocab = text.vocab
        self.train_dataloader = data.BucketIterator(self.train_ds, batch_size)
        self.val_dataloader = data.BucketIterator(self.val_ds, batch_size, repeat=False)
        self.test_dataloader = data.BucketIterator(self.test_ds, batch_size, repeat=False)

    def init_models(self):
        emb_size = self.config['hp'].get('emb_size')
        hid_size = self.config['hp'].get('hid_size')
        latent_size = self.config['hp'].get('latent_size')

        self.rec_criterion = nn.CrossEntropyLoss(size_average=True)
        self.kl_criterion = KLLoss()

        dilations = self.config.get('dilations')
        encoder = LSTMEncoder(emb_size, hid_size, len(self.vocab), latent_size)
        if self.config.get('decoder') == 'LSTM':
            decoder = LSTMDecoder(emb_size, hid_size, len(self.vocab), latent_size)
        else:
            decoder = CNNDecoder(emb_size, hid_size, len(self.vocab), latent_size, dilations=dilations)

        self.vae = cudable(VAE(encoder, decoder, latent_size))
        self.optimizer = Adam(self.vae.parameters(),
                             lr=self.config.get('lr'),
                             betas=self.config.get('adam_betas'))

        self.kl_beta_scheme = HPLinearScheme(*self.config.get('kl_beta_scheme', (0,1,1)))

    def train_on_batch(self, batch):
        rec_loss, kl_loss = self.loss_on_batch(batch)
        kl_beta_coef = compute_param_by_scheme(self.kl_beta_scheme, self.num_iters_done)
        loss = rec_loss + kl_beta_coef * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rec_loss_history.append(rec_loss.item())
        self.kl_loss_history.append(kl_loss.item())

    def loss_on_batch(self, batch):
        batch.text = cudable(batch.text)
        inputs, trg = batch.text[:, :-1], batch.text[:, 1:]
        encodings = self.vae.encoder(inputs)
        means, stds = encodings[:, :self.vae.latent_size], encodings[:, self.vae.latent_size:]
        latents = sample(means, stds)
        recs = self.vae.decoder(means, inputs)

        rec_loss = self.rec_criterion(recs.view(-1, len(self.vocab)), trg.contiguous().view(-1))
        kl_loss = self.kl_criterion(means, stds)

        return rec_loss, kl_loss

    def log_scores(self):
        self.writer.add_scalar('CE loss', self.rec_loss_history[-1], self.num_iters_done)
        self.writer.add_scalar('KL loss', self.kl_loss_history[-1], self.num_iters_done)

    def validate(self):
        rec_losses, kl_losses = [], []

        for batch in self.val_dataloader:
            rec_loss, kl_loss = self.loss_on_batch(batch)
            # rec_ppl = np.exp(min(100, rec_loss.item()))

            # rec_losses.append(rec_ppl)
            rec_losses.append(rec_loss.item())
            kl_losses.append(kl_loss.item())

        self.writer.add_scalar('val_rec_loss', np.mean(rec_losses), self.num_iters_done)
        self.writer.add_scalar('val_kl_loss', np.mean(kl_losses), self.num_iters_done)
        self.test()

    def test(self):
        generated = self.inference(self.test_dataloader)
        originals = [' '.join(e.text).replace('@@ ', '') for e in self.test_ds.examples]
        bleu = compute_bleu_for_sents(generated, originals)
        # text = '\n'.join(['Original: {}\n Generated: {}\n'.format(s,o) for s,o in zip(originals, generated)])
        # Let's wrap in bos/eos so that we see, when empty sequence is generated
        generated = ['[START]' + s + '[END]' for s in generated]
        text = '\n\n'.join(generated)
        self.writer.add_text('Generated examples', text, self.num_iters_done)
        self.writer.add_scalar('Test BLEU', bleu, self.num_iters_done)

    def checkpoint(self):
        self.save_model(self.vae, 'vae')

    def inference(self, dataloader):
        """
        Produces predictions for a given dataset
        """
        seqs = []
        for batch in dataloader:
            inputs = cudable(batch.text[:, :-1])
            seqs.extend(self.vae.inference(inputs, self.vocab))

        return itos_many(seqs, self.vocab)
