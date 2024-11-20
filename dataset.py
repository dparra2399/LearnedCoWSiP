#### Standard Library Imports
import os
from glob import glob

#### Library imports
import numpy as np
import torch
import torch.utils.data
from IRF_layers import Gaussian1DLayer, IRF1DLayer
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from utils.torch_utils import *



from IPython.core import debugger
breakpoint = debugger.set_trace

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, nt, num_samples, photon_counts, sbr, tau, sigma=1, transform=None):

        self.num_samples = num_samples
        if type(photon_counts) == int:
            photon_counts = torch.Tensor([photon_counts] * num_samples)
        if type(sbr) == float:
            sbr = torch.Tensor([sbr] * num_samples)

        assert type(photon_counts) == torch.Tensor and photon_counts.shape[-1] == num_samples
        assert type(sbr) == torch.Tensor and sbr.shape[-1] == num_samples

        self.photon_counts = photon_counts
        self.sbr = sbr
        self.n_tbins = nt
        self.tau = tau

        inputs2D = torch.rand((num_samples, 1), requires_grad=True)

        # Eval Network Components
        model = Gaussian1DLayer(gauss_len=nt)
        outputs = model(inputs2D)

        pulse_domain = np.arange(0, nt)
        pulse = gaussian_pulse(pulse_domain, mu=pulse_domain[-1] // 2, width=sigma, circ_shifted=True)
        irf_layer = IRF1DLayer(irf=pulse, conv_dim=0)
        outputs_irf_layer = irf_layer(outputs)

        scaled_tensors = []
        for i in range(self.num_samples):
            photon_count = self.photon_counts[i]
            sbr = self.sbr[i]
            amb_count = photon_count / sbr
            slice = outputs_irf_layer[i, ...]
            current_area = slice.sum(dim=-1, keepdim=True)
            current_area = torch.where(current_area == 0, torch.ones_like(current_area), current_area)
            amb_per_bin = amb_count / nt
            scaling_factor = photon_count / current_area
            scaled_tensor = slice * scaling_factor + amb_per_bin
            scaled_tensors.append(scaled_tensor)

        self.simulated_data = torch.stack(scaled_tensors, dim=0).unsqueeze(-1)
        self.noisy_data = torch.poisson(self.simulated_data).unsqueeze(-1)
        self.transform = transform


    def __len__(self):
        # Returns the size of the dataset
        return len(self.simulated_data)


    def __getitem__(self, idx):
        # Load data and get label
        noisy_sample = self.noisy_data[idx].squeeze(-1)
        simulated_sample = self.simulated_data[idx]
        gt_depth = simulated_sample.argmax(dim=-2, keepdim=True).squeeze().float()
        #gt_depth = bin2depth(gt_depth, num_bins=self.n_tbins, tau=self.tau)
        # Apply any transformations
        if self.transform:
            noisy_sample = self.transform(noisy_sample)

        return {
            'noisy_sample': noisy_sample,
            'simulated_sample': simulated_sample,
            'gt_depth': gt_depth,
            'idx' : idx
        }
