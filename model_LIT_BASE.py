import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from utils.torch_utils import *
from CM1DLayers import ZNCCLayer
from IPython.core import debugger
breakpoint = debugger.set_trace

class LITBaseModel(pl.LightningModule):

    def __init__(self, backbone_net,
                    k=3, n_tbins=1024,
                    init_lr = 1e-4,
		            lr_decay_gamma = 0.9,
		            loss_id = 'rmse',):
        super(LITBaseModel, self).__init__()

        self.init_lr = init_lr
        self.lr_decay_gamma = lr_decay_gamma
        self.loss_id = loss_id
        self.k = k
        self.n_tbins = n_tbins

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

    def training_step(self, sample, batch_idx):
        # Forward pass
        predicted_depth = self.forward_wrapper(sample)
        # Compute Losses
        loss = self.compute_losses(sample, predicted_depth)
        loss.backward(retain_graph=True)

        # Optionally, you can do manual optimizer step here
        optimizer = self.optimizers()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    def validation_step(self, sample, batch_idx):
        # Forward pass
        predicted_depth = self.forward_wrapper(sample)
        # Compute Losses
        loss = self.compute_losses(sample, predicted_depth)
        # Important NOTE: Newer version of lightning accumulate the val_loss for each batch and then take the mean at the end of the epoch
        self.log_dict({"loss/avg_val": loss})
        # Return depths
        target_depth = sample['gt_depth']
        return {'depth': target_depth, 'depth_predict': predicted_depth}