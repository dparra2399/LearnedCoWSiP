import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from model_LIT_CODING import LITCodingModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.model_utils import SimulatedDataModule
import torch
import pytorch_lightning as pl


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

if_plot = True
rep_freq = 5 * 1e6
rep_tau = 1. / rep_freq
num_samples = 2048
batch_size = 32
sigma = 10

counts = torch.linspace(10 ** 2, 10 ** 4, 10)
sbr = torch.linspace(0.1, 5.0, 10)

init_lr = 0.0001
lr_decay_gamma = 0.95
n_tbins = 1024
k = 4
epochs = 50
beta = 100

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",  # Directory to save the model
    filename="coded_model",  # Base name for the checkpoint files
    save_top_k=1,  # Save only the best model
    monitor="val_loss",  # Metric to monitor
    mode="min",  # Minimize the monitored metric
)

data_module = SimulatedDataModule(n_tbins, counts, sbr, rep_tau, batch_size, num_samples=num_samples, sigma=sigma, normalize=True)
data_module.setup()

pl.seed_everything(42)

logger = CSVLogger("tb_logs", name="my_model")

trainer = pl.Trainer(logger=logger, max_epochs=epochs,
                      log_every_n_steps=50, val_check_interval=0.25,
                      callbacks=[checkpoint_callback])

lit_model = LITCodingModel(k=k, n_tbins=n_tbins, init_lr=init_lr, lr_decay_gamma=lr_decay_gamma, beta=beta)
torch.autograd.set_detect_anomaly(True)

trainer.fit(lit_model, datamodule=data_module)