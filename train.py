import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from dataset import SampleDataset
from torch.utils.data import DataLoader
from model_LIT_BASE import LITBaseModel
from model_LIT_CODING import LITCodingModel
import torch
import pytorch_lightning as pl


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

if_plot = True
rep_freq = 5 * 1e6
rep_tau = 1. / rep_freq
num_samples = 10_000
batch_size = 32
sigma = 10
counts = 10 ** 3
sbr = 1.0
n_tbins = 1024
k = 4
epochs = 5

train_dataset = SampleDataset(n_tbins, num_samples, counts, sbr, sigma=sigma, tau=rep_tau, transform=None)
val_dataset = SampleDataset(n_tbins, num_samples, counts, sbr, sigma=sigma, tau=rep_tau, transform=None)

train_dataloader = DataLoader(train_dataset, shuffle=True)
val_dataloader = DataLoader(val_dataset)

pl.seed_everything(42)

trainer = pl.Trainer(max_epochs=epochs,
                      log_every_n_steps=10, val_check_interval=0.25
                      )

backbone_net = LITCodingModel(k=k, n_tbins=n_tbins)
lit_model = LITBaseModel(backbone_net, k=k, n_tbins=n_tbins)
torch.autograd.set_detect_anomaly(True)

trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)