import torch
import torch.nn as nn
from utils.torch_utils import zero_norm_t
import numpy as np
from utils.tirf_utils import get_coding_scheme
from models.model_LIT_CODING import LITCodingModel, LITIlluminationModel
from models.IRF_layers import Gaussian1DLayer
from IPython.core import debugger
breakpoint = debugger.set_trace

class IlluminationLayer(nn.Module):
    def __init__(self, k=3, n_tbins=1024, init='TruncatedFourier', h_irf=None, get_from_model=False):
        super(IlluminationLayer, self).__init__()
        self.n_tbins = n_tbins
        self.k = k
        self.h_irf = h_irf

        if get_from_model is True:
            model = LITIlluminationModel.load_from_checkpoint(init)
            self.illumination = model.backbone_net.learnable_input.data.detach().cpu()
            self.cmat_init = model.backbone_net.coding_model.cmat1D.weight.data.detach().cpu().numpy().squeeze()
        else:
            cmat_init = get_coding_scheme(coding_id=init, n_tbins=self.n_tbins, k=self.k, h_irf=self.h_irf)
            ill = Gaussian1DLayer(gauss_len=self.n_tbins)
            self.illumination = ill(torch.tensor([0])).detach().cpu().transpose(0, 1)
            self.cmat_init = cmat_init.transpose()

        self.cmat1D = torch.nn.Conv1d(in_channels=self.n_tbins
                                      , out_channels=self.k
                                      , kernel_size=1
                                      , stride=1, padding=0, dilation=1, bias=False)

        self.cmat1D.weight.data = torch.from_numpy(self.cmat_init[..., np.newaxis].astype(np.float32))

    def forward(self, labels, photon_count, sbr):
        current_area = self.illumination.sum(dim=0, keepdim=True)
        amb_count = photon_count / sbr

        amb_per_bin = amb_count / self.n_tbins
        scaling_factor = photon_count / current_area

        scaled_input = self.illumination * scaling_factor + amb_per_bin

        shifted_input = torch.stack([torch.roll(scaled_input, shifts=int(shift), dims=0) for shift in labels],
                                    dim=0)

        #shifted_input[shifted_input < 0] = 0
        noisy_input = torch.poisson(shifted_input)

        c_vals = self.cmat1D(noisy_input)
        return c_vals

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
