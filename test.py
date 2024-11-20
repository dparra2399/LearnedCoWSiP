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
import pytorch_lightning as pl



import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

if_plot = True

rep_freq = 5 * 1e6
rep_tau = 1. / rep_freq
num_samples = 100
sigma = 10
counts = 10 ** 3
sbr = 1.0
n_tbins = 1024
k = 8

sample_data = SampleDataset(n_tbins, num_samples, counts, sbr, sigma=sigma, transform=None)
sample = sample_data.__getitem__(np.random.randint(0, num_samples))

coding_mat = CorrelationMatrixLayer(k=k, n_tbins=n_tbins)
zncc_layer = ZNCCLayer(coding_mat)

c_vals = coding_mat(sample['noisy_sample'])
zncc = zncc_layer(c_vals)

gt_depths = sample['simulate'].argmax(dim=-2, keepdim=True)
gt_depths = bin2depth(gt_depths, num_bins=n_tbins, tau=rep_tau)
pred_depths = zncc.argmax(dim=-2, keepdims=True)
pred_depths = bin2depth(pred_depths, num_bins=n_tbins, tau=rep_tau)

loss_rmse = criterion_RMSE(pred_depths.squeeze(), gt_depths.squeeze())
print(f'RMSE: {loss_rmse * 1000:.3f} mm ')

if if_plot:
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(sample['noisy_sample'], label='Noisy Pulse')
    axs[0].set_title('Noisy Pulse')
    cmat = coding_mat.cmat1D.weight
    cmat = cmat.detach().numpy()
    axs[1].imshow(cmat.squeeze(), aspect='auto', cmap='grey')
    axs[1].set_title('Coding Matrix')
    plt.show()
print('hello world')
