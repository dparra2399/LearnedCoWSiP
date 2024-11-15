import torch
import torch.nn as nn
import torch.nn.init as init
from IPython.core import debugger
breakpoint = debugger.set_trace

class ZNCCLoss(nn.Module):
    def __init__(self):
        super(ZNCCLoss, self).__init__()

    def forward(self, input_hist, target):