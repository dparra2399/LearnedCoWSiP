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
        out = self.backbone_net(x)
        return out

    def forward_wrapper(self, sample):
        input_data = sample['noisy_sample']
        reconstruction = self(input_data)
        return reconstruction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.init_lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay_gamma)
            , 'name': 'epoch/Adam-lr'  # Name for logging in tensorboard (used by lr_monitor callback)
        }
        return [optimizer], [lr_scheduler]

    def compute_losses(self, sample, predicted_depth):
        target_depth = sample['gt_depth']

        loss = criterion_RMSE(predicted_depth, target_depth)
        return loss

    def tv_regularization(self):

        tv_loss = 0.0
        weights = self.backbone_net.cmat1D.weight
        # tv_out = torch.pow(weights[1:, :, :] - weights[:-1, :, :], 2)
        # tv_loss += torch.sum(torch.sqrt(tv_out + 1e-6))

        # TV across input channels
        tv_in = torch.pow(weights[:, 1:, :] - weights[:, :-1, :], 2)
        tv_loss += torch.sum(torch.sqrt(tv_in + 1e-6))
        return tv_loss


    def training_step(self, sample, batch_idx):
        # Forward pass
        predicted_depth = self.forward_wrapper(sample)
        # Compute Losses
        tv_loss = self.tv_regularization()

        loss = self.compute_losses(sample, predicted_depth) + self.tv_reg * tv_loss
        loss.backward(retain_graph=True)

        if (batch_idx % 2000 == 0):
            self.log('train_loss', loss)
            self.log('epoch', self.current_epoch)  # Log current epoch
            self.log('batch', batch_idx)  # Log current batch index

        # Optionally, you can do manual optimizer step here
        optimizer = self.optimizers()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    def validation_step(self, sample, batch_idx):
        # Forward pass
        predicted_depth = self.forward_wrapper(sample)
        # Compute Losses
        tv_loss = self.tv_regularization()

        loss = self.compute_losses(sample, predicted_depth) + self.tv_reg * tv_loss
        # Important NOTE: Newer version of lightning accumulate the val_loss for each batch and then take the mean at the end of the epoch
        if (batch_idx % 2000 == 0):
            self.log_dict({"val_loss": loss})
        # self.log('train_loss', loss)
        # self.log('epoch', self.current_epoch)  # Log current epoch
        # self.log('batch', batch_idx)  # Log current batch index
        # Return depths
        target_depth = sample['gt_depth']
        return {'depth': target_depth, 'depth_predict': predicted_depth}

class LITIlluminationBaseModel(pl.LightningModule):

    def __init__(self, backbone_net,
                    k=3, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',
                    tv_reg = 0.1,
                    photon_count=1e3,
                    sbr=0.1):
        super(LITIlluminationBaseModel, self).__init__()

        self.init_lr = init_lr
        self.lr_decay_gamma = lr_decay_gamma
        self.loss_id = loss_id
        self.k = k
        self.n_tbins = n_tbins
        self.tv_reg = tv_reg
        self.print_logger = logging.getLogger(__name__)
        self.photon_count = photon_count
        self.sbr = sbr
        self.backbone_net = backbone_net
        self.automatic_optimization = False

    def compute_losses(self, sample, predicted_depth):
        target_depth = sample.to(torch.float32)

        loss = criterion_RMSE(predicted_depth, target_depth)
        return loss

    def forward(self, x):
        # use forward for inference/predictions
        out = self.backbone_net(x)
        return out

    def forward_wrapper(self, label):
        reconstruction = self(label)
        return reconstruction

    def training_step(self, sample, batch_idx):
        # Forward pass
        predicted_depth = self.forward_wrapper(sample)
        # Compute Losses
        #tv_loss = self.tv_regularization()

        loss = self.compute_losses(sample, predicted_depth).to(torch.float32)#+ self.tv_reg * tv_loss
        loss.backward(retain_graph=True)

        if (batch_idx % 2000 == 0):
            self.log('train_loss', loss)
            self.log('epoch', self.current_epoch)  # Log current epoch
            self.log('batch', batch_idx)  # Log current batch index

        optimizer = self.optimizers()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    def validation_step(self, sample, batch_idx):
        # Forward pass
        predicted_depth = self.forward_wrapper(sample)
        # Compute Losses
        #tv_loss = self.tv_regularization()

        loss = self.compute_losses(sample, predicted_depth) #+ self.tv_reg * tv_loss
        # Important NOTE: Newer version of lightning accumulate the val_loss for each batch and then take the mean at the end of the epoch
        if (batch_idx % 2000 == 0):
            self.log_dict({"val_loss": loss})
        # self.log('train_loss', loss)
        # self.log('epoch', self.current_epoch)  # Log current epoch
        # self.log('batch', batch_idx)  # Log current batch index
        # Return depths
        return {'depth': sample, 'depth_predict': predicted_depth}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.init_lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay_gamma)
            , 'name': 'epoch/Adam-lr'  # Name for logging in tensorboard (used by lr_monitor callback)
        }
        return [optimizer], [lr_scheduler]
