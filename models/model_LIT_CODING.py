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
import torch.nn.utils as utils
from utils.tirf_utils import get_coding_scheme
breakpoint = debugger.set_trace

class ZNCCCodingModel(nn.Module):
        def __init__(self, k=3, n_tbins=1024, beta=100, init_coding_mat=None, learn_coding_mat=True, load_from_model=False):
            super(ZNCCCodingModel, self).__init__()

            self.k = k
            self.n_tbins = n_tbins
            self.cmat1D = torch.nn.Conv1d(in_channels=self.n_tbins
                                          , out_channels=self.k
                                          , kernel_size=1
                                          , stride=1, padding=0, dilation=1, bias=False)

            for param in self.parameters():
                param.requires_grad = False

            for param in self.cmat1D.parameters():
                param.requires_grad = learn_coding_mat

            self.zero_norm_t = zero_norm_t
            self.beta = beta

            #self.cmat1D.weight.data = self.corr_mat.unsqueeze(-1)
            if init_coding_mat is not None: 
                cmat_init = get_coding_scheme(coding_id=init_coding_mat, n_tbins=self.n_tbins, k=self.k, h_irf=None)
                self.cmat_init = cmat_init.transpose()
                self.cmat1D.weight.data = torch.from_numpy(self.cmat_init[..., np.newaxis].astype(np.float32))
            else:
                init.kaiming_uniform_(self.cmat1D.weight)

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
        

class NCCCodingModel(nn.Module):
        def __init__(self, k=3, n_tbins=1024, beta=100, init_coding_mat=None, learn_coding_mat=True):
            super(NCCCodingModel, self).__init__()

            self.k = k
            self.n_tbins = n_tbins
            self.cmat1D = torch.nn.Conv1d(in_channels=self.n_tbins
                                          , out_channels=self.k
                                          , kernel_size=1
                                          , stride=1, padding=0, dilation=1, bias=False)

            for param in self.parameters():
                param.requires_grad = False

            for param in self.cmat1D.parameters():
                param.requires_grad = learn_coding_mat

            self.norm_t = norm_t
            self.beta = beta

            #self.cmat1D.weight.data = self.corr_mat.unsqueeze(-1)
            if init_coding_mat is not None: 
                cmat_init = get_coding_scheme(coding_id=init_coding_mat, n_tbins=self.n_tbins, k=self.k, h_irf=None)
                self.cmat_init = cmat_init.transpose()
                self.cmat1D.weight.data = torch.from_numpy(self.cmat_init[..., np.newaxis].astype(np.float32))
            else:
                init.kaiming_uniform_(self.cmat1D.weight)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        def softargmax(self, input, dim=-2, beta=100):
            softmaxed = nn.functional.softmax(beta * input, dim=dim)
            indices = torch.arange(self.n_tbins, device=input.device, dtype=torch.float32).view(-1, self.n_tbins, 1)
            result = torch.sum(softmaxed * indices, dim=dim)
            return result


        def forward(self, input_hist):
            # torch.autograd.set_detect_anomaly(True)
            # zero_weight = self.cmat1D.weight - self.cmat1D.weight.mean(dim=0, keepdim=True)
            # tanh_weight = torch.tanh(zero_weight).to(self.device)

            # output = nn.functional.conv1d(
            #     input_hist, zero_weight, self.cmat1D.bias, self.cmat1D.stride, self.cmat1D.padding
            # )
            output = self.cmat1D(input_hist)

            input_norm_t = self.norm_t(output, dim=-2).to(self.device)
            corr_norm_t = self.norm_t(self.cmat1D.weight, dim=0).to(self.device)

            ncc = torch.matmul(torch.transpose(input_norm_t, -2, -1), corr_norm_t.squeeze())
            ncc = torch.transpose(ncc, -2, -1)

            pred_depths = self.softargmax(ncc, dim=-2, beta=self.beta).squeeze(-1).float()

            #print(f"Output grad_fn: {pred_depths.grad_fn}")
            return ncc, pred_depths

class IFFTCodingModel(nn.Module):
        def __init__(self, k=3, n_tbins=1024, beta=100, init_coding_mat=None, learn_coding_mat=True):
            super(IFFTCodingModel, self).__init__()

            self.k = k
            self.n_tbins = n_tbins
            self.cmat1D = torch.nn.Conv1d(in_channels=self.n_tbins
                                          , out_channels=self.k
                                          , kernel_size=1
                                          , stride=1, padding=0, dilation=1, bias=False)

            for param in self.parameters():
                param.requires_grad = False

            for param in self.cmat1D.parameters():
                param.requires_grad = learn_coding_mat

            self.zero_norm_t = zero_norm_t
            self.beta = beta

            if init_coding_mat is not None: 
                cmat_init = get_coding_scheme(coding_id=init_coding_mat, n_tbins=self.n_tbins, k=self.k, h_irf=None)
                self.cmat_init = cmat_init.transpose()
                self.cmat1D.weight.data = torch.from_numpy(self.cmat_init[..., np.newaxis].astype(np.float32))
            else:
                init.kaiming_uniform_(self.cmat1D.weight)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def softargmax(self, input, dim=-2, beta=100):
            softmaxed = nn.functional.softmax(beta * input, dim=dim)
            indices = torch.arange(self.n_tbins, device=input.device, dtype=torch.float32).view(-1, self.n_tbins, 1)
            result = torch.sum(softmaxed * indices, dim=dim)
            return result


        def forward(self, input_hist):
            output = self.cmat1D(input_hist)
            
            phasors = output[:, 0::2, :] - 1j * output[:, 1::2, :]
            phasors = torch.concatenate((torch.zeros(phasors.shape[0], 1, 1, dtype=phasors.dtype).to(self.device), phasors.to(self.device)), dim=-2)
            recon = torch.fft.irfft(phasors, dim=-2, n=self.n_tbins)

            pred_depths = self.softargmax(recon, dim=-2, beta=self.beta).squeeze(-1).float()

            #print(f"Output grad_fn: {pred_depths.grad_fn}")
            return recon, pred_depths

class IlluminationModel(nn.Module):

    def __init__(self, coding_model, k=3, n_tbins=1024, beta=100,  sigma=10, init_illum=None, learn_illum=True):
        super(IlluminationModel, self).__init__()

        self.n_tbins = n_tbins
        self.epilson = 1e-8

        self.learnable_input = nn.Parameter(torch.rand(self.n_tbins, 1), requires_grad=True)

        for param in self.parameters():
            param.requires_grad = False

        self.learnable_input.requires_grad = learn_illum

        if init_illum is not None:
            self.learnable_input.data.fill_(0)
            self.learnable_input.data[0] = 1.0
        else:
            self.learnable_input.data.fill_(1.0) 

        self.coding_model = coding_model
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

        return self.coding_model(noisy_parameter)


class IlluminationPeakModel(nn.Module):

    def __init__(self, coding_model, n_tbins=1024, beta_max=100, peak_factor=5, sigma=10, clamp_peak=True, init_illum=None, learn_illum=True):
        super(IlluminationPeakModel, self).__init__()

        self.n_tbins = n_tbins
        self.epilson = 1e-8

        self.learnable_input = nn.Parameter(torch.rand(self.n_tbins, 1), requires_grad=True)

        for param in self.parameters():
            param.requires_grad = False

        self.learnable_input.requires_grad = learn_illum

        if init_illum is not None:
            self.learnable_input.data.fill_(0)
            self.learnable_input.data[0] = 1.0
        else:
            self.learnable_input.data.fill_(1.0) 

        self.coding_model = coding_model
        pulse_domain = np.arange(0, self.n_tbins)
        pulse = gaussian_pulse(pulse_domain, mu=pulse_domain[-1] // 2, width=sigma, circ_shifted=True)
        self.irf_layer = IRF1DLayer(irf=pulse)

        self.beta_max = beta_max
        self.peak_factor = peak_factor
        self.clamp_peak = clamp_peak

    def smooth_max(self, x, beta=10, dim=0, keepdim=True):
        weights = torch.exp(beta * x)
        return torch.sum(x * weights, dim=dim, keepdim=keepdim) / torch.sum(weights, dim=dim, keepdim=keepdim)
    
    def stable_smooth_max(self, x, beta=10, dim=0, keepdim=True):
        x_max = torch.max(x, dim=dim, keepdim=True).values
        weights = torch.exp(beta * (x - x_max))
        return torch.sum(x * weights, dim=dim, keepdim=keepdim) / torch.sum(weights, dim=dim, keepdim=keepdim)

    def forward(self, bins, photon_counts, sbrs):

        # input = torch.relu(self.learnable_input)
        # irf_input = self.irf_layer(input.view(1, self.n_tbins)).view(self.n_tbins, 1)

        # amb_counts = photon_counts / sbrs  # (batch_size,)
        # amb_per_bin = amb_counts / self.n_tbins  # (batch_size,)

        # current_area = irf_input.sum(dim=0, keepdim=True)  # (n_tbins, 1)
        # scaling_factors = photon_counts.view(-1, 1) / current_area  # (batch_size, 1, 1)

        # shifts = bins.long() % self.n_tbins  # Ensure shifts are valid integers
        # shifted_tensors = torch.stack([torch.roll(irf_input, shifts=int(shift), dims=0) for i, shift in enumerate(shifts)], dim=0)  # (batch_size, n_tbins, 1)

        # scaled_tensors = shifted_tensors * scaling_factors.view(-1, 1, 1) + amb_per_bin.view(-1, 1, 1)  # (batch_size, n_tbins, 1)

        # clamped_tensors = torch.clamp(scaled_tensors, min=None, max=self.peak_factor * photon_counts.view(-1, 1, 1)) 

        # noise = torch.normal(mean=0, std=torch.sqrt(clamped_tensors)).to(clamped_tensors.device)
        # noisy_parameter = clamped_tensors + noise

        
        input = self.irf_layer(torch.relu(self.learnable_input).view(1, self.n_tbins)).view(self.n_tbins, 1)

        shifts = bins.long() % self.n_tbins
        duplicated_tensors = torch.stack([input for i, shift in enumerate(shifts)], dim=0)

        amb_counts = photon_counts / sbrs  # (batch_size,)
        amb_per_bin = amb_counts / self.n_tbins  # (batch_size,)

        current_area = input.sum(dim=0, keepdim=True)  # (n_tbins, 1)
        scaling_factors = photon_counts.view(-1, 1, 1) / current_area  # (batch_size, 1)

        scaled_tensors = duplicated_tensors * scaling_factors.view(-1, 1, 1) # (batch_size, n_tbins, 1)

        if self.clamp_peak:
            clamped_tensors = torch.clamp(scaled_tensors, min=None, max=self.peak_factor * photon_counts.view(-1, 1, 1)) 

            offset_tensors = clamped_tensors  + amb_per_bin.view(-1, 1, 1) 
        else:
            offset_tensors = scaled_tensors + amb_per_bin.view(-1, 1, 1)

        shifted_tensors = torch.stack([torch.roll(self.irf_layer(offset_tensors[i].view(1, self.n_tbins)).view(self.n_tbins, 1) 
                                                  , shifts=int(shift), dims=0) for i, shift in enumerate(shifts)], dim=0)  # (batch_size, n_tbins, 1)
        
        noise = torch.normal(mean=0, std=torch.sqrt(shifted_tensors)).to(shifted_tensors.device)
        noisy_parameter = shifted_tensors + noise
        
        return self.coding_model(noisy_parameter)

class LITIlluminationModel(LITIlluminationBaseModel):
    def __init__(self, k=4, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    beta=100,
                    tv_reg=0.1,
                    sigma=10,
                    recon='zncc',
                    init_coding_mat=None,
                    learn_coding_mat=True,
                    init_illum=None,
                    learn_illum=True):

        if recon=='zncc':
            coding_model = ZNCCCodingModel( k=k, n_tbins=n_tbins, beta=beta, init_coding_mat=init_coding_mat, learn_coding_mat=learn_coding_mat)
        elif recon=='ncc':
            coding_model = NCCCodingModel( k=k, n_tbins=n_tbins, beta=beta, init_coding_mat=init_coding_mat, learn_coding_mat=learn_coding_mat)
        elif recon=='ifft':
            coding_model = IFFTCodingModel( k=k, n_tbins=n_tbins, beta=beta, init_coding_mat=init_coding_mat, learn_coding_mat=learn_coding_mat)
        else:
            assert False, 'no no'

        base_model = IlluminationModel(coding_model, k=k, 
                                       n_tbins=n_tbins, 
                                       sigma=sigma, 
                                       init_illum=init_illum, 
                                       learn_illum=learn_illum)

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
                    beta_max=100,
                    tv_reg=0.1,
                    sigma=10,
                    recon='zncc',
                    peak_factor = 5,
                    peak_reg = 0, 
                    init_coding_mat=None,
                    learn_coding_mat=True,
                    init_illum=None,
                    learn_illum=True):

        if recon=='zncc':
            print('ZNCC Decoding')
            coding_model = ZNCCCodingModel( k=k, n_tbins=n_tbins, beta=beta, init_coding_mat=init_coding_mat, learn_coding_mat=learn_coding_mat)
        elif recon=='ncc':
            coding_model = NCCCodingModel( k=k, n_tbins=n_tbins, beta=beta, init_coding_mat=init_coding_mat, learn_coding_mat=learn_coding_mat)
        elif recon=='ifft':
            print('IFFT Decoding')
            coding_model = IFFTCodingModel( k=k, n_tbins=n_tbins, beta=beta, init_coding_mat=init_coding_mat, learn_coding_mat=learn_coding_mat)
        else:
            assert False, 'no no'

        self.clamp_peak = peak_reg == None
        if peak_reg == None: peak_reg = 0
        base_model = IlluminationPeakModel(coding_model, n_tbins=n_tbins, 
                                           sigma=sigma, 
                                           beta_max=beta_max, 
                                           peak_factor=peak_factor, 
                                           clamp_peak=self.clamp_peak,
                                           init_illum=init_illum, 
                                           learn_illum=learn_illum)

        super(LITIlluminationPeakModel, self).__init__(base_model, k=k, n_tbins=n_tbins,
                                            init_lr = init_lr,
		                                    lr_decay_gamma = lr_decay_gamma,
		                                    loss_id = loss_id,
                                            tv_reg = tv_reg,
                                            peak_factor = peak_factor,
                                            peak_reg=peak_reg)
        self.save_hyperparameters()


class LITCodingModel(LITCodingBaseModel):
    def __init__(self, k=3, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    beta=100,
                    tv_reg=0.1,
                    recon='zncc',
                    init_coding_mat=None,
                    learn_coding_mat=True,):

        if recon=='zncc':
            base_model = ZNCCCodingModel( k=k, n_tbins=n_tbins, beta=beta, init_coding_mat=init_coding_mat, learn_coding_mat=learn_coding_mat)
        elif recon=='ifft':
            base_model = IFFTCodingModel( k=k, n_tbins=n_tbins, beta=beta, init_coding_mat=init_coding_mat, learn_coding_mat=learn_coding_mat)

        super(LITCodingModel, self).__init__(base_model, k=k, n_tbins=n_tbins,
                                            init_lr = init_lr,
		                                    lr_decay_gamma = lr_decay_gamma,
		                                    loss_id = loss_id,
                                            tv_reg = tv_reg,)
        self.save_hyperparameters()



