import torch
import torch.nn as nn
import torch.nn.init as init
from IPython.core import debugger
breakpoint = debugger.set_trace

class BasicModel(nn.Module):

    def __init__(self, k=3, n_tbins=1024):
        super(BasicModel, self).__init__()
        self.n_codes = k
        self.n_tbins = n_tbins
        self.fc1 = nn.Linear(n_tbins, n_tbins * 2)  # Increased layer size for more capacity
        self.fc2 = nn.Linear(n_tbins * 2, n_tbins * 3)

    def forward(self, input_hist):
        input_hist = torch.relu(self.fc1(input_hist))  # Apply ReLU activation
        filter_weights = self.fc2(input_hist)  # Get the flattened filter weights

        # Reshape the output to the desired 2D filter size
        filter_weights = filter_weights.view(-1, 1, self.n_tbins, self.n_codes)
        return filter_weights