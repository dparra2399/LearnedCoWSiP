import logging
import os

from dataset.dataset import SampleDataset, SampleLabels
from torch.utils.data import DataLoader
import numpy as np

import pytorch_lightning as pl
from torch.utils.data import random_split


class SimulatedDataModule(pl.LightningDataModule):
    def __init__(self, nt, photon_counts, sbrs, batch_size, sigma=1,
                 num_samples=None, start_bin=3, end_bin=3,
                 train_val_split: float = 0.8, normalize=False):

        super().__init__()
        self.batch_size = batch_size
        self.n_tbins = nt
        self.photon_counts = photon_counts
        self.sbrs = sbrs
        self.sigma = sigma
        self.normalize = normalize
        if num_samples is None:
            num_samples = self.n_tbins
        self.num_samples = num_samples
        self.train_val_split = train_val_split


    def setup(self, stage=None):
        dataset = SampleDataset(self.n_tbins, self.photon_counts, self.sbrs,
                                    num_samples=self.num_samples, sigma=self.sigma, normalize=self.normalize)

        num_samples = len(dataset)
        train_size = int(np.floor(num_samples * self.train_val_split))
        val_size = int(np.ceil(num_samples * (1 - self.train_val_split)))

        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])


    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4,
                                      persistent_workers=True, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4,
                                    persistent_workers=True)
        return val_dataloader



class SimulatedLabelModule(pl.LightningDataModule):
    def __init__(self, nt, sources, sbrs, batch_size=8,
                 num_samples=None, start_bin=3, end_bin=3,
                 train_val_split: float = 0.8):

        super().__init__()
        self.batch_size = batch_size
        self.n_tbins = nt
        if num_samples is None:
            num_samples = self.n_tbins
        self.num_samples = num_samples
        self.train_val_split = train_val_split
        self.sources = sources
        self.sbrs = sbrs
        self.start_bin = start_bin
        self.end_bin = end_bin

    def setup(self, stage=None):
        dataset = SampleLabels(self.n_tbins, self.sources, self.sbrs, num_samples=self.num_samples,
                               start_bin=self.start_bin, end_bin=self.end_bin)

        num_samples = len(dataset)
        train_size = int(np.floor(num_samples * self.train_val_split))
        val_size = int(np.ceil(num_samples * (1 - self.train_val_split)))

        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])


    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=15,
                                      persistent_workers=True, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15,
                                    persistent_workers=True)
        return val_dataloader