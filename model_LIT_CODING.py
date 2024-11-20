import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils.torch_utils import norm_t, zero_norm_t
from IPython.core import debugger
breakpoint = debugger.set_trace

class LITCodingModel(nn.Module):
        def __init__(self, k=3, n_tbins=1024):
            super(LITCodingModel, self).__init__()
            self.k = k
            self.n_tbins = n_tbins
            #self.corr_mat = nn.Parameter(torch.randn(k, n_tbins), requires_grad=True)
            self.cmat1D = torch.nn.Conv1d(in_channels=self.n_tbins
                                          , out_channels=self.k
                                          , kernel_size=1
                                          , stride=1, padding=0, dilation=1, bias=False)

            self.zero_norm_t = zero_norm_t
            #self.cmat1D.weight.data = self.corr_mat.unsqueeze(-1)



        def forward(self, input_hist):
            output = self.cmat1D(input_hist)[:, 0, :].squeeze()
            # input_norm_t = self.zero_norm_t(output, dim=-2)
            # corr_norm_t = self.zero_norm_t(self.cmat1D.weight.data.detach().clone(), dim=0)
            #
            # zncc = torch.matmul(torch.transpose(input_norm_t, -2, -1), corr_norm_t.squeeze())
            # zncc = torch.transpose(zncc, -2, -1)
            # pred_depths = zncc[:, 0, :]#.argmax(dim=-2).squeeze().double()
            #
            print(f"Output grad_fn: {output.grad_fn}")
            return output
