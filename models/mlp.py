import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error

def get_device():
    device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
    )
    return device

class MLP(nn.Module):
    def __init__(self, input_size, model_order):
        super().__init__()
        hidden_size = (2*input_size + 1)
        # The output size is 2 times the model order (len of poles) for the residues,
        # and 2 again since each coeff is a complex value and we expect to return 0im for the real-only components. 
        output_size = model_order * 2 * 2
        self.layers = nn.Sequential(
            # Input layer
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # First hidden layer (Fully Connected)
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # Output layer
            nn.Linear(hidden_size, output_size),
        )
        # If run into performance issues try converting data instead of model
        self.double()
        # Model convention is to return: real residues, real poles, complex reisudes, complex poles

    def forward(self, x):
        output = self.layers(x)
        complex_output = torch.view_as_complex(output.view(-1, 2))
        return complex_output

    def loss_fn(self, actual_S, predicted_S):
        return mean_absolute_percentage_error(actual_S, predicted_S)

