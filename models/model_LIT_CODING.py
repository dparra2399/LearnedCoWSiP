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

            init.kaiming_uniform_(self.cmat1D.weight)
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
            corr_norm_t = self.zero_norm_t(self.cmat1D.weight, dim=0)

            zncc = torch.matmul(torch.transpose(input_norm_t, -2, -1), corr_norm_t.squeeze())
            zncc = torch.transpose(zncc, -2, -1)
            pred_depths = self.softargmax(zncc, dim=-2, beta=self.beta).squeeze(-1).float()

            #print(f"Output grad_fn: {pred_depths.grad_fn}")
            return zncc, pred_depths


class IlluminationModel(nn.Module):

    def __init__(self, k=3, n_tbins=1024, beta=100,  sigma=10, normalize=False):
        super(IlluminationModel, self).__init__()

        self.n_tbins = n_tbins
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

    def forward(self, bins, photon_counts, sbrs):

        input = torch.relu(self.learnable_input)
        irf_input = self.irf_layer(input.view(1, self.n_tbins)).view(self.n_tbins, 1)

        amb_counts = photon_counts / sbrs  # (batch_size,)
        amb_per_bin = amb_counts / self.n_tbins  # (batch_size,)

        current_area = irf_input.sum(dim=0, keepdim=True)  # (n_tbins, 1)
        scaling_factors = photon_counts.view(-1, 1) / current_area  # (batch_size, 1, 1)

        shifts = bins.long() % self.n_tbins  # Ensure shifts are valid integers
        shifted_tensors = torch.stack([torch.roll(irf_input, shifts=int(shift), dims=0) for i, shift in enumerate(shifts)], dim=0)  # (batch_size, n_tbins, 1)

        scaled_tensors = shifted_tensors * scaling_factors.view(-1, 1, 1) + amb_per_bin.view(-1, 1, 1)  # (batch_size, n_tbins, 1)

        noise = torch.normal(mean=0, std=torch.sqrt(scaled_tensors)).to(scaled_tensors.device)
        noisy_parameter = scaled_tensors + noise

        #input_min = noisy_parameter.min(dim=1, keepdim=True)[0]
        #input_max = noisy_parameter.max(dim=1, keepdim=True)[0]
        #normal_input = (noisy_parameter - input_min) / (input_max - input_min + self.epilson)
        return self.coding_model(noisy_parameter)


class IlluminationPeakModel(nn.Module):

    def __init__(self, k=3, n_tbins=1024, beta=100,  sigma=10, normalize=False):
        super(IlluminationPeakModel, self).__init__()

        self.n_tbins = n_tbins
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

    def forward(self, bins, peak_counts, ambient_counts):

        input = torch.relu(self.learnable_input)
        irf_input = self.irf_layer(input.view(1, self.n_tbins)).view(self.n_tbins, 1)

        current_peak = torch.logsumexp(irf_input, dim=0, keepdim=True) # (n_tbins, 1)

        shifts = bins.long() % self.n_tbins  # Ensure shifts are valid integers
        shifted_tensors = torch.stack([torch.roll(irf_input, shifts=int(shift), dims=0) for i, shift in enumerate(shifts)], dim=0)  # (batch_size, n_tbins, 1)

        scaled_tensors = ((shifted_tensors / current_peak) * peak_counts.view(-1, 1, 1)) + ambient_counts.view(-1, 1, 1)  # (batch_size, n_tbins, 1)

        noise = torch.normal(mean=0, std=torch.sqrt(scaled_tensors)).to(scaled_tensors.device)
        noisy_parameter = scaled_tensors + noise

        #input_min = noisy_parameter.min(dim=1, keepdim=True)[0]
        #input_max = noisy_parameter.max(dim=1, keepdim=True)[0]
        #normal_input = (noisy_parameter - input_min) / (input_max - input_min + self.epilson)
        return self.coding_model(noisy_parameter)


class LITIlluminationModel(LITIlluminationBaseModel):
    def __init__(self, k=4, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    beta=100,
                    tv_reg=0.1,
                    sigma=10):

        base_model = IlluminationModel(k=k, n_tbins=n_tbins, beta=beta, sigma=sigma)
        super(LITIlluminationModel, self).__init__(base_model, k=k, n_tbins=n_tbins,
                                            init_lr = init_lr,
		                                    lr_decay_gamma = lr_decay_gamma,
		                                    loss_id = loss_id,
                                            tv_reg = tv_reg,)
        self.save_hyperparameters()


class LITIlluminationPeakModel(LITIlluminationBaseModel):
    def __init__(self, k=4, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    beta=100,
                    tv_reg=0.1,
                    sigma=10):

        base_model = IlluminationModel(k=k, n_tbins=n_tbins, beta=beta, sigma=sigma)
        super(LITIlluminationPeakModel, self).__init__(base_model, k=k, n_tbins=n_tbins,
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



