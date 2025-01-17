import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils.torch_utils import norm_t, zero_norm_t
from IPython.core import debugger
from models.model_LIT_BASE import LITCodingBaseModel, LITIlluminationBaseModel
from models.IRF_layers import IRF1DLayer
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
import torch.nn.init as init
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

            init.xavier_uniform_(self.cmat1D.weight)
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


class IlluminationModel(nn.Module):

    def __init__(self, k=3, n_tbins=1024, beta=100, photon_count=1e3, sbr=0.1, sigma=10, normalize=False):
        super(IlluminationModel, self).__init__()

        self.n_tbins = n_tbins
        self.photon_count = photon_count
        self.sbr = sbr
        self.epilson = 1e-8

        self.learnable_input = nn.Parameter(torch.rand(self.n_tbins, 1), requires_grad=True)

        for param in self.parameters():
            param.requires_grad = False

        self.learnable_input.requires_grad = True
        self.learnable_input.data.fill_(1.0) 

        self.coding_model = CodingModel(k=k, n_tbins=n_tbins, beta=beta)
        pulse_domain = np.arange(0, self.n_tbins)
        pulse = gaussian_pulse(pulse_domain, mu=pulse_domain[-1] // 2, width=sigma, circ_shifted=True)
        self.irf_layer = IRF1DLayer(irf=pulse)

    def forward(self, bins):

        irf_output = self.irf_layer(self.learnable_input.view(1, self.n_tbins)).view(self.n_tbins, 1)
        current_area = irf_output.sum(dim=0, keepdim=True)
        current_area = torch.where(current_area == 0, torch.ones_like(current_area), current_area)
        amb_count = self.photon_count / self.sbr

        amb_per_bin = amb_count / self.n_tbins
        scaling_factor = self.photon_count / current_area

        scaled_input = irf_output * scaling_factor + amb_per_bin

        shifted_input = torch.stack([torch.roll(scaled_input, shifts=int(shift), dims=0) for shift in bins],
                                    dim=0)

        #input_min = shifted_input.min(dim=0, keepdim=True)[0]
        #input_max = shifted_input.max(dim=0, keepdim=True)[0]

        #normal_shifted_input = (shifted_input - input_min) / (input_max - input_min + self.epilson)
        #normal_shifted_input = torch.where((input_max - input_min) < self.epilson, torch.full_like(normal_shifted_input, self.epilson), normal_shifted_input)

        #normal_shifted_input = (shifted_input - shifted_input.mean(dim=0, keepdim=True)) / (shifted_input.std(dim=0, keepdim=True) + self.epilsonepilson)

        shifted_input = torch.relu(shifted_input)
        
        # noise = torch.poisson(shifted_input)
        # noisy_parameter = shifted_input + noise

        # noisy_parameter = noisy_parameter.detach() + noisy_parameter - noisy_parameter.detach()

        noise = torch.normal(mean=0, std=torch.sqrt(shifted_input)).to(shifted_input.device)
        noisy_parameter = shifted_input + noise

        return self.coding_model(noisy_parameter)



class LITIlluminationModel(LITIlluminationBaseModel):
    def __init__(self, k=4, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    beta=100,
                    tv_reg=0.1,
                    photon_count=1e3,
                    sbr=0.1,
                    sigma=10):

        base_model = IlluminationModel(k=k, n_tbins=n_tbins, beta=beta, photon_count=photon_count, sbr=sbr, sigma=sigma)
        super(LITIlluminationModel, self).__init__(base_model, k=k, n_tbins=n_tbins,
                                            init_lr = init_lr,
		                                    lr_decay_gamma = lr_decay_gamma,
		                                    loss_id = loss_id,
                                            tv_reg = tv_reg,)
        self.save_hyperparameters()


class LITCodingModel(LITCodingBaseModel):
    def __init__(self, k=3, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    beta=100,
                    tv_reg=0.1):

        base_model = CodingModel(k=k, n_tbins=n_tbins, beta=beta)
        super(LITCodingModel, self).__init__(base_model, k=k, n_tbins=n_tbins,
                                            init_lr = init_lr,
		                                    lr_decay_gamma = lr_decay_gamma,
		                                    loss_id = loss_id,
                                            tv_reg = tv_reg,)
        self.save_hyperparameters()



