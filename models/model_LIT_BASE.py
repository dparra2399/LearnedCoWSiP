import torch
import torch.nn as nn
import torch.nn.init as init
import logging

import pytorch_lightning as pl
from utils.torch_utils import *
from IPython.core import debugger
breakpoint = debugger.set_trace

class LITCodingBaseModel(pl.LightningModule):

    def __init__(self, backbone_net,
                    k=3, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    tv_reg = 0.1):
        super(LITCodingBaseModel, self).__init__()

        self.init_lr = init_lr
        self.lr_decay_gamma = lr_decay_gamma
        self.loss_id = loss_id
        self.k = k
        self.n_tbins = n_tbins
        self.tv_reg = tv_reg
        self.print_logger = logging.getLogger(__name__)

        self.backbone_net = backbone_net
        self.automatic_optimization = False


    def forward(self, x):
        # use forward for inference/predictions
        reconstruction, p_depth = self.backbone_net(x)
        return reconstruction, p_depth

    def forward_wrapper(self, sample):
        input_data = sample['noisy_sample']
        reconstruction, p_depth = self(input_data)
        return reconstruction, p_depth

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.init_lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay_gamma)
            , 'name': 'epoch/Adam-lr'  # Name for logging in tensorboard (used by lr_monitor callback)
        }
        return [optimizer], [lr_scheduler]

    def compute_losses(self, sample, reconstruction, predicted_depth):
        target_depth = sample['gt_depth']

        if self.loss_id == 'rmse':
            loss = criterion_RMSE(predicted_depth, target_depth)
        else:
            loss = criterion_RMSE(predicted_depth, target_depth)
        return loss

    def tv_regularization(self):

        tv_loss = 0.0
        weights = self.backbone_net.cmat1D.weight
        tv_in = torch.pow(weights[:, 1:, :] - weights[:, :-1, :], 2)
        tv_loss += torch.sum(torch.sqrt(tv_in + 1e-6))
        return tv_loss


    def training_step(self, sample, batch_idx):
        recon, p_depth = self.forward_wrapper(sample)


        tv_loss = self.tv_regularization()

        loss = self.compute_losses(sample, recon, p_depth).to(torch.float32) + self.tv_reg * tv_loss

        if (batch_idx % 300 == 0):
            self.log('train_loss', loss)
            self.log('epoch', self.current_epoch)  # Log current epoch
            self.log('batch', batch_idx)  # Log current batch index

        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        optimizer.step()
        return loss

    def validation_step(self, sample, batch_idx):
        recon, p_depth = self.forward_wrapper(sample)

        tv_loss = self.tv_regularization()

        loss = self.compute_losses(sample, recon, p_depth).to(torch.float32) + self.tv_reg * tv_loss

        if (batch_idx % 250 == 0):
            self.log_dict({"val_loss": loss})

        return {'depth': sample['gt_depth'], 'depth_predict': p_depth, 'reconstruction': recon}

class LITIlluminationBaseModel(pl.LightningModule):

    def __init__(self, backbone_net,
                    k=3, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    tv_reg = 0.1):
        super(LITIlluminationBaseModel, self).__init__()

        self.init_lr = init_lr
        self.lr_decay_gamma = lr_decay_gamma
        self.loss_id = loss_id
        self.k = k
        self.n_tbins = n_tbins
        self.tv_reg = tv_reg
        self.print_logger = logging.getLogger(__name__)
        self.backbone_net = backbone_net
        self.automatic_optimization = False

    def tv_regularization(self):
        tv_loss = 0.0
        weights = self.backbone_net.coding_model.cmat1D.weight
        tv_in = torch.pow(weights[:, 1:, :] - weights[:, :-1, :], 2)
        tv_loss += torch.sum(torch.sqrt(tv_in + 1e-6))
        return tv_loss
    
    def compute_losses(self, sample, reconstruction, predicted_depth):
        target_depth = sample['depth'].to(torch.float32)

        if self.loss_id == 'rmse':
            loss = criterion_RMSE(predicted_depth, target_depth)
        elif self.loss_id == 'crossentorpy':
            loss = nn.CrossEntropyLoss(reconstruction, target_depth)
        else:
            loss = criterion_RMSE(predicted_depth, target_depth)
        return loss

    def forward(self, depth, photon_count, sbr):
        # use forward for inference/predictions
        recon, p_depth = self.backbone_net(depth, photon_count, sbr)
        return recon, p_depth

    def forward_wrapper(self, sample):
        depth = sample['depth']
        photon_count = sample['photon_count']
        sbr = sample['sbr']
        recon, p_depth = self(depth, photon_count, sbr)
        return  recon, p_depth

    def training_step(self, sample, batch_idx):
        recon, p_depth = self.forward_wrapper(sample)


        tv_loss = self.tv_regularization()

        loss = self.compute_losses(sample, recon, p_depth).to(torch.float32) + self.tv_reg * tv_loss

        if (batch_idx % 300 == 0):
            self.log('train_loss', loss)
            self.log('epoch', self.current_epoch)  # Log current epoch
            self.log('batch', batch_idx)  # Log current batch index

        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        optimizer.step()
        return loss

    def validation_step(self, sample, batch_idx):
        recon, p_depth = self.forward_wrapper(sample)

        tv_loss = self.tv_regularization()

        loss = self.compute_losses(sample, recon, p_depth).to(torch.float32) + self.tv_reg * tv_loss

        if (batch_idx % 250 == 0):
            self.log_dict({"val_loss": loss})

        return {'depth': sample['depth'], 'depth_predict': p_depth, 'reconstruction': recon}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.init_lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay_gamma)
            , 'name': 'epoch/Adam-lr'  # Name for logging in tensorboard (used by lr_monitor callback)
        }
        return [optimizer], [lr_scheduler]
