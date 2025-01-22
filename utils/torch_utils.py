from IPython.core import debugger
import torch
import torch.nn as nn
import os

C = 3e8

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

def norm_t(signal, dim=-2):
    mn = torch.mean(signal, dim=dim, keepdim=True)
    return signal - mn

def zero_norm_t(signal, dim=-2):
    norm_sig = norm_t(signal, dim=dim)
    std = torch.std(signal, dim=dim, keepdim=True)
    return norm_sig / std


def criterion_RMSE(est, gt):
    criterion = nn.MSELoss()
    # est should have grad
    return torch.sqrt(criterion(est, gt))

