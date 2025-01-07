import logging
import os

import numpy as np
from model_LIT_CODING import LITCodingModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.model_utils import SimulatedDataModule
import torch
import pytorch_lightning as pl


import matplotlib.pyplot as plt
import matplotlib
import yaml
#matplotlib.use('TkAgg')

if_plot = True
rep_freq = 5 * 1e6
rep_tau = 1. / rep_freq
num_samples = 1024
batch_size = 64
sigma = 10

counts = torch.linspace(10 ** 2, 10 ** 6, 10)
sbr = torch.linspace(0.1, 10.0, 10)
#
init_lr = 0.001
lr_decay_gamma = 0.9
tv_reg = 0.01
n_tbins = 1024
k = 4
epochs = 100
beta = 10

yaml_file = 'best_hyperparameters_200.yaml'

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # Directory to save the model
        filename="coded_model",  # Base name for the checkpoint files
        save_top_k=1,  # Save only the best model
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Minimize the monitored metric
    )

    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        init_lr = config['init_lr']
        lr_decay_gamma = config['lr_decay_gamma']
        tv_reg = config['tv_reg']
        epochs = config['epochs']
        batch_size = config['batch_size']
        beta = config['beta']
        num_samples = config['num_samples']
    except (FileNotFoundError, TypeError) as e:
        pass

    data_module = SimulatedDataModule(n_tbins, counts, sbr, rep_tau, batch_size, num_samples=num_samples, sigma=sigma, normalize=True)
    data_module.setup()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_float32_matmul_precision('medium')

    else:
        device = torch.device("cpu")

    pl.seed_everything(42)

    logger = CSVLogger("tb_logs", name="my_model")

    trainer = pl.Trainer(logger=logger, max_epochs=epochs,
                          log_every_n_steps=250, val_check_interval=0.50,
                          callbacks=[checkpoint_callback])

    lit_model = LITCodingModel(k=k, n_tbins=n_tbins, init_lr=init_lr, lr_decay_gamma=lr_decay_gamma,
                               beta=beta, tv_reg=tv_reg)
    torch.autograd.set_detect_anomaly(True)

    trainer.fit(lit_model, datamodule=data_module)