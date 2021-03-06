import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.nn.utils import clip_grad_norm_
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseTrainer
from firelab.utils import cudable, HPLinearScheme, compute_param_by_scheme

from src.models import VAE, RNNEncoder, RNNDecoder, CNNDecoder
from src.models.vae import sample
from src.losses import KLLoss, compute_bleu_for_sents
from src.utils.common import itos_many


class VAETrainer(BaseTrainer):
    def __init__(self, config):
        super(VAETrainer, self).__init__(config)

    def init_dataloaders(self):
        batch_size = self.config.get('batch_size', 8)
        project_path = self.config['firelab']['project_path']
        data_path = os.path.join(project_path, self.config['data'])

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
        self.train_dataloader = data.BucketIterator(self.train_ds, batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(self.val_ds, batch_size, train=False, sort=False)
        self.test_dataloader = data.BucketIterator(self.test_ds, batch_size, train=False, sort=False)

    def init_models(self):
        emb_size = self.config['hp'].get('emb_size')
        hid_size = self.config['hp'].get('hid_size')

        weights = cudable(torch.ones(len(self.vocab)))
        weights[self.vocab.stoi['<pad>']] = 0
        self.rec_criterion = nn.CrossEntropyLoss(weights, size_average=True)
        self.kl_criterion = KLLoss()

        encoder = RNNEncoder(emb_size, hid_size, len(self.vocab))

        if self.config.get('decoder') == 'RNN':
            decoder = RNNDecoder(emb_size, hid_size, len(self.vocab))
        else:
            dilations = self.config.get('dilations')
            decoder = CNNDecoder(emb_size, hid_size, len(self.vocab), hid_size, dilations=dilations)

        self.vae = cudable(VAE(encoder, decoder, hid_size))
        self.optimizer = self.construct_optimizer()
        self.desired_kl_val = HPLinearScheme(*self.config.get('desired_kl_val', (0,0,1)))
        self.force_kl = HPLinearScheme(*self.config.get('force_kl', (1,1,1)))
        self.decoder_dropword_scheme = HPLinearScheme(*self.config.get('decoder_dropword_scheme', (0,0,1)))
        self.noiseness_scheme = HPLinearScheme(*self.config.get('noiseness_scheme', (1,1,1)))

        self.try_to_load_checkpoint()

    def train_on_batch(self, batch):
        self.train_mode()

        # Computing losses
        desired_kl = compute_param_by_scheme(self.desired_kl_val, self.num_iters_done)
        force_kl = compute_param_by_scheme(self.force_kl, self.num_iters_done)

        rec_loss, kl_loss, (means, log_stds) = self.loss_on_batch(batch)
        loss = rec_loss + force_kl * abs(kl_loss - desired_kl)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = math.sqrt((sum([w.grad.norm()**2 for w in self.vae.parameters()])))
        weights_norm = math.sqrt((sum([w.norm()**2 for w in self.vae.parameters()])))
        weights_l_inf_norm = max([w.abs().max() for w in self.vae.parameters()])

        if 'grad_clip' in self.config['hp']:
            clip_grad_norm_(self.vae.parameters(), self.config['hp']['grad_clip'])

        self.optimizer.step()

        # Logging stuff
        self.writer.add_scalar('Total loss', loss, self.num_iters_done)
        self.writer.add_scalar('CE loss', rec_loss, self.num_iters_done)
        self.writer.add_scalar('KL loss', kl_loss, self.num_iters_done)
        self.writer.add_scalar('Desired KL', desired_kl, self.num_iters_done)
        self.writer.add_scalar('Force KL', force_kl, self.num_iters_done)
        self.writer.add_scalar('Means norm', means.norm(dim=1).mean(), self.num_iters_done)
        self.writer.add_scalar('Stds norm', log_stds.exp().norm(dim=1).mean(), self.num_iters_done)
        self.writer.add_scalar('Grad norm', grad_norm, self.num_iters_done)
        self.writer.add_scalar('Weights norm', weights_norm, self.num_iters_done)
        self.writer.add_scalar('Weights l_inf norm', weights_l_inf_norm, self.num_iters_done)

    def loss_on_batch(self, batch):
        noiseness = compute_param_by_scheme(self.noiseness_scheme, self.num_iters_done)
        dropword_p = compute_param_by_scheme(self.decoder_dropword_scheme, self.num_iters_done)

        batch.text = cudable(batch.text)
        (means, log_stds), predictions = self.vae(batch.text, noiseness, dropword_p)

        rec_loss = self.rec_criterion(predictions.view(-1, len(self.vocab)),
                                      batch.text[:, 1:].contiguous().view(-1))
        kl_loss = self.kl_criterion(means, log_stds.exp())

        return rec_loss, kl_loss, (means, log_stds)

    def validate(self):
        self.eval_mode()

        rec_losses, kl_losses = [], []

        for batch in self.val_dataloader:
            rec_loss, kl_loss, _ = self.loss_on_batch(batch)
            rec_ppl = np.exp(min(100, rec_loss.item()))

            rec_losses.append(rec_ppl)
            rec_losses.append(rec_loss.item())
            kl_losses.append(kl_loss.item())

        self.writer.add_scalar('val_rec_loss', np.mean(rec_losses), self.num_iters_done)
        self.writer.add_scalar('val_kl_loss', np.mean(kl_losses), self.num_iters_done)
        self.validate_predictions()

    def validate_predictions(self):
        """
        Performs inference on a val dataloader
        (computes predictions without teacher's forcing)
        """
        generated, originals = self.inference(self.val_dataloader)
        bleu = compute_bleu_for_sents(generated, originals)
        generated = ['[{}] => [{}]'.format(o,g) for o,g in zip(originals, generated)]
        text = '\n\n'.join(generated)

        self.writer.add_text('Generated examples', text, self.num_iters_done)
        self.writer.add_scalar('Validation BLEU', bleu, self.num_iters_done)

    def checkpoint(self):
        self.save_module_state(self.vae, 'vae')
        self.save_module_state(self.optimizer, 'optimizer')

    def try_to_load_checkpoint(self):
        """
        Loads model state from checkpoint if it is provided
        """
        if not 'continue_from_iter' in self.config['firelab']: return

        self.num_iters_done = self.config['firelab'].get('continue_from_iter')
        self.num_epochs_done = self.num_iters_done // len(self.train_dataloader)

        self.load_module_state(self.vae, 'vae', self.num_iters_done)
        self.load_module_state(self.optimizer, 'optimizer', self.num_iters_done)

    def inference(self, dataloader):
        """
        Produces predictions for a given dataloader
        """
        seqs = []
        originals = []
        noiseness = compute_param_by_scheme(self.noiseness_scheme, self.num_iters_done)

        for batch in dataloader:
            inputs = cudable(batch.text)
            seqs.extend(self.vae.inference(inputs, self.vocab, noiseness))
            originals.extend(inputs.detach().cpu().numpy().tolist())

        return itos_many(seqs, self.vocab), itos_many(originals, self.vocab)

    def construct_optimizer(self):
        """
        Builds an optimizer according to config arguments
        """
        weight_decay = self.config.get('weight_decay', 0)

        if self.config.get('optimizer') == 'RMSProp':
            return RMSprop(self.vae.parameters(), lr=self.config.get('lr'),
                           weight_decay=weight_decay)
        else:
            return Adam(self.vae.parameters(), lr=self.config.get('lr'),
                        betas=self.config.get('adam_betas', (0.9, 0.999)),
                        weight_decay=weight_decay)

    def train_mode(self):
        self.vae.train()

    def eval_mode(self):
        self.vae.eval()
