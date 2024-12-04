import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from dataset import SampleDataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class SimulatedDataModule(pl.LightningDataModule):
    def __init__(self, nt, photon_counts, sbr, tau, batch_size, sigma=1,
                 num_samples=None, train_val_split: float = 0.8, normalize=False):

        super().__init__()
        self.batch_size = batch_size
        self.n_tbins = nt
        self.photon_counts = photon_counts
        self.sbr = sbr
        self.tau = tau
        self.sigma = sigma
        self.normalize = normalize
        if num_samples is None:
            num_samples = self.n_tbins
        self.num_samples = num_samples
        self.train_val_split = train_val_split

    def setup(self, stage=None):
        train_size = self.num_samples * self.train_val_split
        val_size = self.num_samples * (1 - self.train_val_split)
        self.train_dataset = SampleDataset(self.n_tbins, self.photon_counts, self.sbr,
                                      num_samples=train_size, sigma=self.sigma, tau=self.tau, normalize=self.normalize)
        self.val_dataset = SampleDataset(self.n_tbins, self.photon_counts, self.sbr,
                                      num_samples=val_size, sigma=self.sigma, tau=self.tau, normalize=self.normalize)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        return val_dataloader

