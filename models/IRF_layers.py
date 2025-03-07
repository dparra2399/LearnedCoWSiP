from random import gauss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core import debugger
breakpoint = debugger.set_trace

from felipe_utils.research_utils.signalproc_ops import gaussian_pulse



class Gaussian1DLayer(nn.Module):
	'''
		For each input mu (assume 0 < mu < 1) we generate a 1D gaussian signal with mean=mu and unit sigma
	'''
	def __init__(self, gauss_len=1024):
		super(Gaussian1DLayer, self).__init__()

		self.gauss_len = gauss_len
		self.sigma = 1. / self.gauss_len
		self.sigma_sq = self.sigma ** 2
		# Normalization factor
		self.a = (1. / (self.sigma * np.sqrt(2*np.pi))) / self.gauss_len
		domain = torch.arange(0, self.gauss_len, requires_grad=False) / self.gauss_len
		self.domain = torch.nn.Parameter(domain, requires_grad=False)

	def forward(self, mu):
		loc_reshaped = mu.unsqueeze(-1) - self.domain
		out_gaussian = self.a*torch.exp(-0.5*torch.square(loc_reshaped / self.sigma))
		return out_gaussian


class IRF1DLayer(nn.Module):

    def __init__(self, irf, conv_dim=0):
        super(IRF1DLayer, self).__init__()

        assert (irf.ndim == 1), "Input IRF should have only 1 dimension"

        self.n = irf.size  # Number of bins in irf
        self.is_even = int((self.n % 2) == 0)
        self.pad_l = self.n // 2
        self.pad_r = self.n // 2 - self.is_even
        self.conv_padding = (self.pad_l, self.pad_r)
        # Normalize
        irf = irf / irf.sum()
        # Flip irf vector
        irf = np.flip(irf)

        irf_tensor = torch.from_numpy(irf.reshape((1, 1, self.n)).astype(np.float32))
        self.irf_weights = torch.nn.Parameter(irf_tensor, requires_grad=False)

    def forward(self, inputs):
        '''
            Inputs should have dims (B, 1, D0, D1, D2)
        '''
        # Pad inputs appropriately so that the convolution is the same as a circular convolution
        padded_input = F.pad(inputs, pad=self.conv_padding, mode='circular')
        # Apply convolution
        # No need to pad here since we padded above to make sure that the out has the same dimension
        out = F.conv1d(padded_input, self.irf_weights, stride=1, padding=0)
        return out.squeeze(1)
