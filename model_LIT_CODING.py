import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils.torch_utils import norm_t, zero_norm_t
from IPython.core import debugger
from model_LIT_BASE import LITBaseModel
breakpoint = debugger.set_trace

class CodingModel(nn.Module):
        def __init__(self, k=3, n_tbins=1024, beta=100):
            super(CodingModel, self).__init__()

            self.k = k
            self.n_tbins = n_tbins
            self.cmat1D = torch.nn.Conv1d(in_channels=self.n_tbins
                                          , out_channels=self.k
                                          , kernel_size=1
                                          , stride=1, padding=0, dilation=1, bias=False)

            for param in self.parameters():
                param.requires_grad = False

            # Make only the Conv1d layer trainable
            for param in self.cmat1D.parameters():
                param.requires_grad = True


            self.zero_norm_t = zero_norm_t
            self.beta = beta

            #self.cmat1D.weight.data = self.corr_mat.unsqueeze(-1)

        def softargmax(self, input, dim=-2, beta=100):
            softmaxed = nn.functional.softmax(beta * input, dim=dim)
            indices = torch.arange(self.n_tbins, device=input.device, dtype=torch.float32).view(-1, self.n_tbins, 1)
            result = torch.sum(softmaxed * indices, dim=dim)
            return result

        def forward(self, input_hist):
            output = self.cmat1D(input_hist)
            input_norm_t = self.zero_norm_t(output, dim=-2)
            corr_norm_t = self.zero_norm_t(self.cmat1D.weight.data.detach().clone(), dim=0)

            zncc = torch.matmul(torch.transpose(input_norm_t, -2, -1), corr_norm_t.squeeze())
            zncc = torch.transpose(zncc, -2, -1)
            pred_depths = self.softargmax(zncc, dim=-2, beta=self.beta).squeeze(-1).float()

            #print(f"Output grad_fn: {pred_depths.grad_fn}")
            return pred_depths


class LITCodingModel(LITBaseModel):
    def __init__(self, k=3, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    beta=100):

        base_model = CodingModel(k=k, n_tbins=n_tbins, beta=beta)
        super(LITCodingModel, self).__init__(base_model, k=k, n_tbins=n_tbins,
                                            init_lr = init_lr,
		                                    lr_decay_gamma = lr_decay_gamma,
		                                    loss_id = loss_id,)
        self.save_hyperparameters()

