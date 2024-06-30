import torch
import torch.nn as nn
import numpy as np
import unittest
import matplotlib.pyplot as mplt

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_features, scale=10):
        super(FourierFeatures, self).__init__()
        # Creating the B matrix
        self.B = nn.Parameter(torch.randn(size=(num_features, input_dim)) * scale, requires_grad=False)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Projecting input x to a higher-dimensional space using B
        x_proj = 2 * np.pi * torch.matmul(x, self.B.T)
        # Concatenating cos and sin transformations
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class TestFourierFeatures(unittest.TestCase):
    def test_fourier_features_output(self):
        input_dim = 2
        num_features = 10
        scale = 10
        ff = FourierFeatures(input_dim, num_features, scale)
        
        # Creating test inputs
        test_inputs = torch.tensor([[0.1, 0.2], [10.3, 100.4]])
        outputs = ff(test_inputs)

        # Check the output dimensions
        self.assertEqual(outputs.shape, (2, 2 * num_features))  # Check output size
        
        # Check output range for sine and cosine values
        self.assertTrue(torch.all((outputs >= -1) & (outputs <= 1)))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
