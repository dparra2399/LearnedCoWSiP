#### Standard Library Imports

#### Library imports
import numpy as np
import torch.utils.data
from models.IRF_layers import Gaussian1DLayer, IRF1DLayer
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from utils.torch_utils import *

from IPython.core import debugger
breakpoint = debugger.set_trace

class SampleLabels(torch.utils.data.Dataset):
    def __init__(self, nt, photon_counts, sbrs, num_samples=10):
        self.nt = nt
        self.num_samples = num_samples
        labels = torch.linspace(3, self.nt-3, self.num_samples).to(torch.int)
        
        labeled_tensors = []
        for i in range(photon_counts.shape[-1]):
            photon_count = photon_counts[i]
            for j in range(sbrs.shape[-1]):
                sbr = sbrs[j]
                for k in range(labels.shape[-1]):
                    label = labels[k]
                    tup = (label, photon_count, sbr)
                    labeled_tensors.append(tup)
                
        self.labels = torch.tensor(labeled_tensors)
        #self.labels = self.labels.view(self.labels.shape[-1], -1)

    def __len__(self):
        # Returns the size of the dataset
        return self.labels.shape[0]

    def __getitem__(self, idx):
        sample = self.labels[idx]
        label = sample[0]
        photon_count = sample[1]
        sbr = sample[2]
        return {'depth': label, 'photon_count': photon_count, 'sbr': sbr}

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, nt, photon_counts, sbr, num_samples=None, sigma=10, normalize=False):

        if type(photon_counts) == int:
            photon_counts = torch.Tensor([photon_counts])
        if type(sbr) == float:
            sbr = torch.Tensor([sbr])

        assert type(photon_counts) == torch.Tensor
        assert type(sbr) == torch.Tensor

        self.photon_counts = photon_counts
        self.sbr = sbr
        self.n_tbins = nt

        if num_samples is None:
            num_samples = 1
        self.num_samples = int(num_samples)

        inputs1D = torch.linspace(3, self.n_tbins-3, self.num_samples) / self.n_tbins
        inputs1D = inputs1D.view(inputs1D.shape[-1], -1)
        model = Gaussian1DLayer(gauss_len=nt)
        outputs = model(inputs1D)

        pulse_domain = np.arange(0, self.n_tbins)
        pulse = gaussian_pulse(pulse_domain, mu=pulse_domain[-1] // 2, width=sigma, circ_shifted=True)
        irf_layer = IRF1DLayer(irf=pulse, conv_dim=0)
        outputs_irf_layer = irf_layer(outputs)

        scaled_tensors = []
        for i in range(self.photon_counts.shape[-1]):
            photon_count = self.photon_counts[i]
            for j in range(self.sbr.shape[-1]):
                sbr = self.sbr[j]
                amb_count = photon_count / sbr
                slice = outputs_irf_layer
                current_area = slice.sum(dim=-1, keepdim=True)
                current_area = torch.where(current_area == 0, torch.ones_like(current_area), current_area)
                amb_per_bin = amb_count / nt
                scaling_factor = photon_count / current_area
                scaled_tensor = slice * scaling_factor + amb_per_bin
                scaled_tensors.extend(scaled_tensor)

        self.simulated_data = torch.stack(scaled_tensors, dim=0).unsqueeze(-1)
        self.noisy_data = torch.poisson(self.simulated_data)
        self.normalize = normalize
        if self.normalize:
            num_samples = len(self)
            mean_noisy = self.noisy_data.mean(dim=-2, keepdim=True)
            std_noisy = self.noisy_data.std(dim=-2, keepdim=True)
            self.noisy_data = (self.noisy_data - mean_noisy) / std_noisy
            mean = self.simulated_data.mean(dim=-2, keepdim=True)
            std = self.simulated_data.std(dim=-2, keepdim=True)
            self.simulated_data = (self.simulated_data - mean) / std

    def __len__(self):
        # Returns the size of the dataset
        return self.simulated_data.shape[0]


    def __getitem__(self, idx):
        # Load data and get label
        noisy_sample = self.noisy_data[idx]
        simulated_sample = self.simulated_data[idx]
        gt_depth = simulated_sample.argmax(dim=-2, keepdim=True).squeeze().float()
        # Apply any transformations

        return {
            'noisy_sample': noisy_sample,
            'simulated_sample': simulated_sample,
            'gt_depth': gt_depth,
            'idx' : idx
        }
