import torch
import torch.nn as nn
from utils.torch_utils import zero_norm_t, norm_t
import numpy as np
from utils.tirf_utils import get_coding_scheme
from models.model_LIT_CODING import LITCodingModel, LITIlluminationModel
from models.IRF_layers import Gaussian1DLayer, IRF1DLayer
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from IPython.core import debugger
breakpoint = debugger.set_trace

class IlluminationLayer(nn.Module):
    def __init__(self, k=3, n_tbins=1024, init='TruncatedFourier', sigma=10, h_irf=None, get_from_model=False):
        super(IlluminationLayer, self).__init__()
        self.n_tbins = n_tbins
        self.k = k
        self.h_irf = h_irf

        if get_from_model is True:
            model = LITIlluminationModel.load_from_checkpoint(init, strict=False)
            self.illumination = model.backbone_net.learnable_input.data.detach().cpu().view(self.n_tbins, 1)
            self.cmat_init = model.backbone_net.coding_model.cmat1D.weight.data.detach().cpu().numpy().squeeze()
        else:
            cmat_init = get_coding_scheme(coding_id=init, n_tbins=self.n_tbins, k=self.k, h_irf=self.h_irf)
            ill = Gaussian1DLayer(gauss_len=self.n_tbins)
            self.illumination = ill(torch.tensor([0])).detach().cpu().view(self.n_tbins, 1)
            self.cmat_init = cmat_init.transpose()

        self.cmat1D = torch.nn.Conv1d(in_channels=self.n_tbins
                                      , out_channels=self.k
                                      , kernel_size=1
                                      , stride=1, padding=0, dilation=1, bias=False)

        self.cmat1D.weight.data = torch.from_numpy(self.cmat_init[..., np.newaxis].astype(np.float32))

        pulse_domain = np.arange(0, self.n_tbins)
        pulse = gaussian_pulse(pulse_domain, mu=pulse_domain[-1] // 2, width=sigma, circ_shifted=True)
        self.irf_layer = IRF1DLayer(irf=pulse)
        
    def forward(self, bins, photon_counts, sbrs):

        #SMOOTHING AFTER AVERAGE POWER CALCULATION : Takes really long....

        input = torch.relu(self.illumination)

        shifts = bins.long() % self.n_tbins
        duplicated_tensors = torch.stack([input for i, shift in enumerate(shifts)], dim=0)

        amb_counts = photon_counts / sbrs  # (batch_size,)
        amb_per_bin = amb_counts / self.n_tbins  # (batch_size,)

        current_area = input.sum(dim=0, keepdim=True)  # (n_tbins, 1)
        scaling_factors = photon_counts.view(-1, 1, 1) / current_area  # (batch_size, 1)

        scaled_tensors = duplicated_tensors * scaling_factors.view(-1, 1, 1) + amb_per_bin.view(-1, 1, 1)  # (batch_size, n_tbins, 1)

        shifted_tensors = torch.stack([torch.roll(self.irf_layer(scaled_tensors[i].view(1, self.n_tbins)).view(self.n_tbins, 1) 
                                                  , shifts=int(shift), dims=0) for i, shift in enumerate(shifts)], dim=0)  # (batch_size, n_tbins, 1)
        
        noisy_input = torch.poisson(shifted_tensors)

        # input = torch.relu(self.illumination)
        # irf_input = self.irf_layer(input.view(1, self.n_tbins)).view(self.n_tbins, 1)

        # amb_counts = photon_counts / sbrs  # (batch_size,)
        # amb_per_bin = amb_counts / self.n_tbins  # (batch_size,)

        # current_area = irf_input.sum(dim=0, keepdim=True)  # (n_tbins, 1)
        # scaling_factors = photon_counts.view(-1, 1) / current_area  # (batch_size, 1, 1)

        # shifts = bins.long() % self.n_tbins  # Ensure shifts are valid integers
        # shifted_tensors = torch.stack([torch.roll(irf_input, shifts=int(shift), dims=0) for i, shift in enumerate(shifts)], dim=0)  # (batch_size, n_tbins, 1)

        # scaled_tensors = shifted_tensors * scaling_factors.view(-1, 1, 1) + amb_per_bin.view(-1, 1, 1)  # (batch_size, n_tbins, 1)
        
        # noisy_input = torch.poisson(scaled_tensors)

        return self.cmat1D(noisy_input)

class IlluminationPeakLayer(nn.Module):
    def __init__(self, k=3, n_tbins=1024, init='TruncatedFourier', sigma=10, peak_factor = 5, h_irf=None, get_from_model=False):
        super(IlluminationPeakLayer, self).__init__()
        self.n_tbins = n_tbins
        self.k = k
        self.h_irf = h_irf

        if get_from_model is True:
            model = LITIlluminationModel.load_from_checkpoint(init, strict=False)
            self.illumination = model.backbone_net.learnable_input.data.detach().cpu().view(1, self.n_tbins)
            self.cmat_init = model.backbone_net.coding_model.cmat1D.weight.data.detach().cpu().numpy().squeeze()
        else:
            cmat_init = get_coding_scheme(coding_id=init, n_tbins=self.n_tbins, k=self.k, h_irf=self.h_irf)
            ill = Gaussian1DLayer(gauss_len=self.n_tbins)
            self.illumination = ill(torch.tensor([0])).detach().cpu().view(1, self.n_tbins)
            self.cmat_init = cmat_init.transpose()

        self.cmat1D = torch.nn.Conv1d(in_channels=self.n_tbins
                                      , out_channels=self.k
                                      , kernel_size=1
                                      , stride=1, padding=0, dilation=1, bias=False)

        self.cmat1D.weight.data = torch.from_numpy(self.cmat_init[..., np.newaxis].astype(np.float32))

        pulse_domain = np.arange(0, self.n_tbins)
        pulse = gaussian_pulse(pulse_domain, mu=pulse_domain[-1] // 2, width=sigma, circ_shifted=True)
        self.irf_layer = IRF1DLayer(irf=pulse)
        self.peak_factor = peak_factor


    def get_output_illumination(self, photon_count, sbr):
        illum = self.irf_layer(self.illumination.view(1, self.n_tbins)).view(self.n_tbins, 1).squeeze()
        illum[illum < 0] = 0

        current_area = illum.sum(dim=0, keepdim=True)  # (n_tbins, 1)
        scaling_factor = photon_count / current_area  # (batch_size, 1, 1)
        
        amb_counts = photon_count / sbr  # (batch_size,)
        amb_per_bin = amb_counts / self.n_tbins  # (batch_size,)

        scaled_tensor = illum * scaling_factor  # (batch_size, n_tbins, 1)
        clamped_tensor = torch.clamp(scaled_tensor, min=None, max=self.peak_factor * photon_count) 

        clamped_tensor = self.irf_layer(clamped_tensor.view(1, self.n_tbins)).view(self.n_tbins, 1)
        clamped_tensor = clamped_tensor + amb_per_bin
        return clamped_tensor

        
    def forward(self, bins, photon_counts, sbrs):  

        input = self.irf_layer(torch.relu(self.illumination).view(1, self.n_tbins)).view(self.n_tbins, 1)

        shifts = bins.long() % self.n_tbins
        duplicated_tensors = torch.stack([input for i, shift in enumerate(shifts)], dim=0)

        amb_counts = photon_counts / sbrs  # (batch_size,)
        amb_per_bin = amb_counts / self.n_tbins  # (batch_size,)

        current_area = input.sum(dim=0, keepdim=True)  # (n_tbins, 1)
        scaling_factors = photon_counts.view(-1, 1, 1) / current_area  # (batch_size, 1)

        scaled_tensors = duplicated_tensors * scaling_factors.view(-1, 1, 1) # (batch_size, n_tbins, 1)

        clamped_tensors = torch.clamp(scaled_tensors, min=None, max=self.peak_factor * photon_counts.view(-1, 1, 1)) 

        offset_tensors = clamped_tensors  + amb_per_bin.view(-1, 1, 1) 

        shifted_tensors = torch.stack([torch.roll(self.irf_layer(offset_tensors[i].view(1, self.n_tbins)).view(self.n_tbins, 1) 
                                                  , shifts=int(shift), dims=0) for i, shift in enumerate(shifts)], dim=0)  # (batch_size, n_tbins, 1)
        
        noisy_input = torch.poisson(shifted_tensors)

        return self.cmat1D(noisy_input)

    
class CodingLayer(nn.Module):
    def __init__(self, k=3, n_tbins=1024, init='TruncatedFourier', h_irf=None, get_from_model=False):
        super(CodingLayer, self).__init__()

        self.n_tbins = n_tbins
        self.k = k

        self.h_irf = h_irf


        if get_from_model:
            model = LITCodingModel.load_from_checkpoint(init)
            self.cmat_init = model.backbone_net.cmat1D.weight.data.detach().cpu().numpy().squeeze()
        else:
            cmat_init = get_coding_scheme(coding_id=init, n_tbins=self.n_tbins, k=self.k, h_irf=self.h_irf)
            self.cmat_init = cmat_init.transpose()


        self.cmat1D = torch.nn.Conv1d(in_channels=self.n_tbins
                                      , out_channels=self.k
                                      , kernel_size=1
                                      , stride=1, padding=0, dilation=1, bias=False)

        self.cmat1D.weight.data = torch.from_numpy(self.cmat_init[..., np.newaxis].astype(np.float32))

    def forward(self, inputs):
        '''
            Expected input dims == (Batch, Nt, num_samples)
        '''
        ## Compute compressive histogram
        c_vals = self.cmat1D(inputs)
        return c_vals

class NCCLayer(nn.Module):
    def __init__(self):
        super(NCCLayer, self).__init__()

        self.norm_t = norm_t

    def forward(self, input_compressed, cmat):
            # Normalize images
        input_norm_t = self.norm_t(input_compressed, dim=-2)
        corr_norm_t = self.norm_t(cmat, dim=0)

        # Calculate cross-correlation
        ncc = torch.matmul(torch.transpose(input_norm_t, -2, -1), corr_norm_t.squeeze())
        pred_depths = torch.argmax(torch.transpose(ncc, -2, -1), dim=-2).squeeze(-1)

        return pred_depths
    
class ZNCCLayer(nn.Module):
    def __init__(self):
        super(ZNCCLayer, self).__init__()

        self.zero_norm_t = zero_norm_t

    def forward(self, input_compressed, cmat):
            # Normalize images
        input_norm_t = self.zero_norm_t(input_compressed, dim=-2)
        corr_norm_t = self.zero_norm_t(cmat, dim=0)

        # Calculate cross-correlation
        zncc = torch.matmul(torch.transpose(input_norm_t, -2, -1), corr_norm_t.squeeze())
        pred_depths = torch.argmax(torch.transpose(zncc, -2, -1), dim=-2).squeeze(-1)

        return pred_depths
    
class IFFTReconLayer(nn.Module):
    def __init__(self, n_tbins):
        self.n_tbins = n_tbins
        super(IFFTReconLayer, self).__init__()


    def forward(self, input_compressed):
        phasors = input_compressed[:, 0::2, :] - 1j * input_compressed[:, 1::2, :]
        #print(phasors.shape)
        phasors = torch.concatenate((torch.zeros(phasors.shape[0], 1, 1, dtype=phasors.dtype), phasors), dim=-2)
        #print(phasors.shape)
        recon = torch.fft.irfft(phasors, dim=-2, n=self.n_tbins)
        #print(recon.shape)
        pred_depths = torch.argmax(recon, dim=-2).squeeze(-1)
        return pred_depths
