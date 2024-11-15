from IPython.core import debugger
import torch
import os

def norm_t(signal, dim=-1):
    mn = torch.mean(signal, dim=dim, keepdim=True)
    return signal - mn

def zero_norm_t(signal, dim=-1):
    norm_sig = norm_t(signal, dim=dim)
    std = torch.std(signal, dim=dim, keepdim=True)
    return norm_sig / std


def zncc(signal1, signal2, dims):
    # Normalize images
    sig1_norm_t = zero_norm_t(signal1)
    sig2_norm_t = zero_norm_t(signal2)

    # Calculate cross-correlation
    corr = torch.nn.functional.conv2d(sig1_norm_t, sig2_norm_t.flip(dims=dims))

    return corr