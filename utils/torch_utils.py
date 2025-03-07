from IPython.core import debugger
import torch
import torch.nn as nn
import os

C = 3e8
EPSILON = 1e-8

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, gt):
        return torch.mean(torch.sqrt((pred - gt) ** 2 + self.epsilon ** 2))


def bin2tof(b, num_bins, tau):
    '''
        b == bin
        num_bins == number of bins in histogram
        tau == period
    '''
    return (b / num_bins) * tau

def tof2depth(tof):
    return tof * C / 2.

def bin2depth(b, num_bins, tau):
    return tof2depth(bin2tof(b, num_bins, tau))

def norm_t(signal, dim=-1):
    return signal / (torch.linalg.norm(signal, ord=2, dim=dim, keepdim=True) + EPSILON)

def zero_norm_t(signal, dim=-2):
    norm_sig = signal - torch.mean(signal, dim=dim, keepdim=True)
    std = torch.std(signal, dim=dim, keepdim=True)
    return norm_sig / std


def criterion_RMSE(est, gt):
    criterion = nn.MSELoss()
    # est should have grad
    return torch.sqrt(criterion(est, gt))

def criterion_MAE(est, gt):
    criterion = nn.L1Loss()
    return criterion(est, gt)

def criterion_CHARBONNIER(est, gt):
    criterion = CharbonnierLoss()
    return criterion(est, gt)
