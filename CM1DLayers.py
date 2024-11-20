import torch
import torch.nn as nn
import torch.nn.init as init
from utils.torch_utils import norm_t, zero_norm_t
import numpy as np
from utils.tirf_utils import get_coding_scheme
from IPython.core import debugger
breakpoint = debugger.set_trace

class CorrelationMatrixLayer(nn.Module):
    def __init__(self, k=3, n_tbins=1024, init='TruncatedFourier', h_irf=None, optimize_mat=False):
        super(CorrelationMatrixLayer, self).__init__()

        self.n_tbins = n_tbins
        self.k = k

        self.h_irf = h_irf


        if optimize_mat:
            self.cmat_init = nn.Parameter(torch.randn(k, n_tbins), requires_grad=True)
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
    def __init__(self, corr_layer : CorrelationMatrixLayer):
        super(ZNCCLayer, self).__init__()

        self.corr_layer = corr_layer
        self.corr_mat = corr_layer.cmat1D.weight
        self.zero_norm_t = zero_norm_t

    def forward(self, input_compressed):
            # Normalize images
        input_norm_t = self.zero_norm_t(input_compressed, dim=-2)
        corr_norm_t = self.zero_norm_t(self.corr_mat, dim=0)

        # Calculate cross-correlation
        zncc = torch.matmul(torch.transpose(input_norm_t, -2, -1), corr_norm_t.squeeze())

        return torch.transpose(zncc, -2, -1)


