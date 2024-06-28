import torch
import torch.nn as nn
import numpy as np

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_features, scale=10):
        super(FourierFeatures, self).__init__()
        # Creating the B matrix
        self.B = nn.Parameter(torch.randn(size=(num_features, input_dim)) * scale, requires_grad=False)

    def forward(self, x):
        # Projecting input x to a higher-dimensional space using B
        x_proj = 2 * np.pi * torch.matmul(x, self.B.T)
        # Concatenating cos and sin transformations
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
