import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseTrainer
from tensorboardX import SummaryWriter

from src.models import VAE
from src.models.vae import sample
from src.losses import KLLoss

use_cuda = torch.cuda.is_available()


class VAETrainer(BaseTrainer):
    def __init__(self, config):
        super(VAETrainer, self).__init__(config)
        self.config = config

        self.rec_loss_history = []
        self.kl_loss_history = []

        # self.rec_loss_file = open(os.path.join(config['firelab']['logs_path'], 'rec_loss.csv'), 'w')
        # self.kl_loss_file = open(os.path.join(config['firelab']['logs_path'], 'kl_loss.csv'), 'w')
        # self.rec_loss_log_path = os.path.join(config['firelab']['logs_path'], 'rec_loss.log')
        # self.kl_loss_log_path = os.path.join(config['firelab']['logs_path'], 'kl_loss.log')

        self.num_iters_done = 0
        self.max_num_epochs = config.get('max_num_epochs', 10)
        self.plots_update_freq = config.get('plots_update_freq', 50)
        self.val_freq = config.get('val_freq', 100)

        # self.vis = visdom.Visdom(port=self.config.get('visdom-port'))
        # self.rec_plot_id = None
        # self.kl_plot_id = None
        self.writer = SummaryWriter(config['firelab']['logs_path'])

    def init_dataloaders(self):
        project_path = self.config['firelab']['project_path']
        data_path = os.path.join(project_path, 'data/yelp-reviews.tok.bpe.random100k')

        with open(data_path) as f: lines = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)
        examples = [Example.fromlist([s], [('text', text)]) for s in lines]
        dataset = Dataset(examples, [('text', text)])
        text.build_vocab(dataset)

        self.vocab = text.vocab
        self.train_dataloader = data.BucketIterator(
            dataset=dataset, batch_size=self.config.get('batch_size', 8))

    def init_models(self):
        emb_size = self.config['hp'].get('emb_size', 256)
        hid_size = self.config['hp'].get('hid_size', 256)
        latent_size = self.config['hp'].get('latent_size', 32)

        self.rec_criterion = nn.CrossEntropyLoss(size_average=True)
        self.kl_criterion = KLLoss()

        self.vae = VAE(emb_size, hid_size, len(self.vocab), latent_size)
        self.optimizer = Adam(self.vae.parameters(), lr=1e-4)

        if use_cuda: self.vae.cuda()

        self.kl_beta_coef = self.config['hp'].get('kl_beta_coef', 0.5)

    def train_on_batch(self, batch):
        inputs, targets = batch.text[:, :-1], batch.text[:, 1:]
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        encodings = self.vae.encoder(inputs)
        means, stds = encodings[:, :self.vae.latent_size], encodings[:, self.vae.latent_size:]
        latents = sample(means, stds)
        reconstructions = self.vae.decoder(latents, inputs)

        rec_loss = self.rec_criterion(reconstructions.view(-1, len(self.vocab)), targets.contiguous().view(-1))
        kl_loss = self.kl_criterion(means, stds)
        loss = rec_loss + self.kl_beta_coef * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rec_loss_history.append(rec_loss.item())
        self.kl_loss_history.append(kl_loss.item())
        self.log_losses()

    def log_losses(self):
        # self.rec_loss_file.write(str(self.rec_loss_history[-1]) + '\n')
        # self.kl_loss_file.write(str(self.kl_loss_history[-1]) + '\n')
        self.writer.add_scalar('rec_loss', self.rec_loss_history[-1], self.num_iters_done)
        self.writer.add_scalar('kl_loss', self.kl_loss_history[-1], self.num_iters_done)

    def start(self):
        self.init_dataloaders()
        self.init_models()
        self.run_training()
        self.close_log_files_io()

    # def inference(self, batch):
    #     inputs, targets = batch.text.transpose(0,1).cuda(), batch.target.transpose(0,1).cuda()
    #     encodings = encoder(inputs)
    #     means = encodings[:, :32]
    #     reconstructions = decoder.generate(means) # Using means instead of sampling
    #     tokens = reconstructions.max(dim=-1)[1]

    def update_plots(self):
        return
        if self.num_iters_done % self.plots_update_freq != 0: return

        n_new_iters = self.plots_update_freq
        n_old_iters = len(self.rec_loss_history) - n_new_iters
        rec_x_vals = np.arange(n_old_iters, n_old_iters + n_new_iters)
        kl_x_vals = np.arange(n_old_iters, n_old_iters + n_new_iters)

        if self.rec_plot_id != None and self.kl_plot_id != None:
            rec_losses = np.array(self.rec_loss_history[-n_new_iters:])
            kl_losses = np.array(self.kl_loss_history[-n_new_iters:])

            self.vis.line(rec_losses, rec_x_vals, update='append', win=self.rec_plot_id)
            self.vis.line(kl_losses, kl_x_vals, update='append', win=self.kl_plot_id)
        else:
            # Drawing from scratch
            self.rec_plot_id = self.vis.line(np.array(self.rec_loss_history), rec_x_vals)
            self.kl_plot_id = self.vis.line(np.array(self.kl_loss_history), kl_x_vals)

        # plt.plot(rec_loss_history)
        # plt.plot(pd.DataFrame(np.array(rec_loss_history)).ewm(span=100).mean())

        # plt.plot(kl_loss_history)
        # plt.plot(pd.DataFrame(np.array(kl_loss_history)).ewm(span=100).mean())

    def validate(self):
        if not self.validate_every or self.num_iters_done % self.validate_every != 0: return

        # val_losses = []
        # for val_batch in val_data:
        #     val_src, val_tgt = val_batch
        #     val_pred = model(val_src, val_tgt)
        #     val_loss = criterion(val_pred, val_tgt[:, 1:].contiguous().view(-1))
        #     val_losses.append(val_loss.data[0])

        # val_loss_history.append(np.mean(val_losses))
        # val_loss_iters.append(num_iters_done)

    # def close_log_files_io(self):
    #     self.rec_loss_file.close()
    #     self.kl_loss_file.close()

    def save_model(self):
        model_name = '{}-{}.pth'.format(self.config['name'], self.num_iters_done)
        model_path = os.path.join(self.config['firelab']['checkpoints_path'], model_name)
        torch.save(self.vae.state_dict(), model_path)
