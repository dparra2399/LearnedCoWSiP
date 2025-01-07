import logging

import os

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

init = 'checkpoints/coded_model-v8.ckpt'

coding_mat = CorrelationMatrixLayer(k=4, n_tbins=1024, init=init, get_from_model=True)

cmat = coding_mat.cmat1D.weight
cmat = np.transpose(cmat.detach().numpy().squeeze())

np.save(init.split('/')[-1], cmat)





