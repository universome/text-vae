import os
import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseTrainer
from firelab.utils import cudable

from src.models import RNNEncoder, RNNDecoder
from src.losses import compute_bleu_for_sents
from src.utils.common import itos_many
from src.models.utils import inference


class AETrainer(BaseTrainer):
    def __init__(self, config):
        super(AETrainer, self).__init__(config)

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
        self.encoder = cudable(RNNEncoder(emb_size, hid_size, len(self.vocab)))
        self.decoder = cudable(RNNDecoder(emb_size, hid_size, len(self.vocab)))

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = Adam(parameters, lr=self.config.get('lr'))

    def train_on_batch(self, batch):
        loss = self.loss_on_batch(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('CE loss', loss, self.num_iters_done)

    def loss_on_batch(self, batch):
        batch.text = cudable(batch.text)
        inputs, trg = batch.text[:, :-1], batch.text[:, 1:]
        encodings = self.encoder(batch.text)
        recs = self.decoder(encodings, inputs)

        rec_loss = self.rec_criterion(recs.view(-1, len(self.vocab)), trg.contiguous().view(-1))

        return rec_loss

    def validate(self):
        rec_losses = []

        for batch in self.val_dataloader:
            rec_losses.append(self.loss_on_batch(batch).item())

        self.writer.add_scalar('val_rec_loss', np.mean(rec_losses), self.num_iters_done)
        self.compute_val_bleu()

    def compute_val_bleu(self):
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

    def inference(self, dataloader):
        """
        Produces predictions for a given dataloader
        """
        seqs = []
        originals = []

        for batch in dataloader:
            inputs = cudable(batch.text)
            encodings = self.encoder(inputs)
            sentences = inference(self.decoder, encodings, self.vocab)

            seqs.extend(sentences)
            originals.extend(inputs.detach().cpu().numpy().tolist())

        return itos_many(seqs, self.vocab), itos_many(originals, self.vocab)
