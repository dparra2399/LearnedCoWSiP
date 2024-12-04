import logging

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pytorch_lightning as pl

import numpy as np
import torch
from utils.torch_utils import bin2depth
from CM1DLayers import CorrelationMatrixLayer, ZNCCLayer
from dataset import SampleDataset
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from utils.torch_utils import *


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

if_plot = True

rep_freq = 5 * 1e6
rep_tau = 1. / rep_freq
sigma = 10
# counts = torch.linspace(10 ** 2, 10 ** 6, 20)
# sbr = torch.linspace(0.05, 10.0, 20)
counts = 10 ** 4
sbr = 1.0
n_tbins = 1024
k = 4

init = 'checkpoints/good_checkpoints/coded_model-v9.ckpt'

sample_data = SampleDataset(n_tbins, counts, sbr, num_samples=5000, tau=rep_tau, sigma=sigma)
num_samples = len(sample_data)
sample = sample_data.noisy_data

coding_mat = CorrelationMatrixLayer(k=k, n_tbins=n_tbins, init=init, get_from_model=True)
coding_mat_base = CorrelationMatrixLayer(k=k, n_tbins=n_tbins)
zncc_layer = ZNCCLayer()

c_vals = coding_mat(sample)
c_vals_base = coding_mat_base(sample)

pred_depths = zncc_layer(c_vals, coding_mat.cmat1D.weight.data.detach().clone())
pred_depths_base = zncc_layer(c_vals_base, coding_mat_base.cmat1D.weight.data.detach().clone())

gt_depths = sample_data.simulated_data.argmax(dim=-2).squeeze(-1)
gt_depths = bin2depth(gt_depths, num_bins=n_tbins, tau=rep_tau)
pred_depths = bin2depth(pred_depths, num_bins=n_tbins, tau=rep_tau)
pred_depths_base = bin2depth(pred_depths_base, num_bins=n_tbins, tau=rep_tau)


loss_rmse = criterion_RMSE(pred_depths.squeeze(), gt_depths.squeeze())
loss_rmse_base = criterion_RMSE(pred_depths_base.squeeze(), gt_depths.squeeze())

print(f'RMSE Learned: {loss_rmse * 1000:.3f} mm ')
print(f'RMSE Trunc. Fourier: {loss_rmse_base * 1000:.3f} mm ')


if if_plot:
    fig, axs = plt.subplots(1, 4)
    axs[0].plot(sample.__getitem__(np.random.randint(0, num_samples)), label='Noisy Pulse')
    axs[0].set_title('Sample Measured Histogram')
    cmat = coding_mat.cmat1D.weight
    cmat = np.transpose(cmat.detach().numpy().squeeze())
    axs[1].plot(cmat)
    axs[1].set_title('Learned')
    cmat2 = coding_mat_base.cmat1D.weight
    cmat2 = np.transpose(cmat2.detach().numpy().squeeze())
    axs[2].plot(cmat2)
    axs[2].set_title('Trunc. Fourier')

    axs[3].bar(0, loss_rmse,  label=f'{loss_rmse * 1000:.2f}mm')
    axs[3].bar(1, loss_rmse_base,  label=f'{loss_rmse_base * 1000:.2f}mm')
    axs[3].set_title('RMSE (Lower Better)')
    axs[3].set_ylabel('RMSE (mm)')
    axs[3].set_xticks([0, 1])
    axs[3].set_xticklabels(['Learned', 'Trunc. Fourier'])
    axs[3].legend()
    plt.show(block=True)
print('hello world')

