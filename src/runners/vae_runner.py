import os

import torch
import torch.nn as nn
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseRunner

from src.models import VAE

use_cuda = torch.cuda.is_available()


class VAERunner(BaseRunner):
    def __init__(self, config):
        self.config = config

    def init_dataloaders(self):
        project_path = self.config['firelab']['project_path']
        data_path = os.path.join(project_path, 'data/yelp-reviews.tok.bpe.random100k')

        with open(data_path) as f: lines = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)
        examples = [Example.fromlist([s], [('text', text)]) for s in lines]
        dataset = Dataset(examples, [('text', text)])
        text.build_vocab(dataset)

        self.vocab = text.vocab
        self.train_dataloader = data.BucketIterator(dataset=dataset, batch_size=32)

    def init_models(self):
        emb_size = config['hp'].get('emb_size', 256)
        hid_size = config['hp'].get('hid_size', 256)
        latent_size = config['hp'].get('latent_size', 32)

        self.rec_criterion = nn.CrossEntropyLoss(size_average=True)
        self.kld_criterion = KLLoss()

        self.vae = VAE(emb_size, hid_size, len(self.vocab), latent_size)
        self.optimizer = Adam(self.vae.parameters(), lr=1e-4)

    def init_trainer(self):
        self.trainer = VAETrainer(self.vae, self.optimizer, self.config, self.on_iter_done)

    def start(self):
        self.init_dataloaders()
        self.init_models()
        self.init_trainer()
        self.trainer.run_training(self.train_dataloader)

    def inference(self, batch):
        inputs, targets = batch.text.transpose(0,1).cuda(), batch.target.transpose(0,1).cuda()
        encodings = encoder(inputs)
        means = encodings[:, :32]
        reconstructions = decoder.generate(means) # Using means instead of sampling
        tokens = reconstructions.max(dim=-1)[1]

    def on_iter_done():
        if should_plot and num_iters_done % 10 == 0:
            plt.title("Reconstruction loss")
            plt.plot(rec_loss_history)
            plt.plot(pd.DataFrame(np.array(rec_loss_history)).ewm(span=100).mean())

            plt.title("KL loss")
            plt.plot(kl_loss_history)
            plt.plot(pd.DataFrame(np.array(kl_loss_history)).ewm(span=100).mean())

        if num_iters_done % 10 == 0 and num_iters_done > 0:
            val_losses = []
            for val_batch in val_data:
                val_src, val_tgt = val_batch
                val_pred = model(val_src, val_tgt)
                val_loss = criterion(val_pred, val_tgt[:, 1:].contiguous().view(-1))
                val_losses.append(val_loss.data[0])

            val_loss_history.append(np.mean(val_losses))
            val_loss_iters.append(num_iters_done)
