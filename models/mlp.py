import torch
import torch.nn as nn
from torchmetrics.functional.regression import mean_absolute_percentage_error

def get_device():
    device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
    )
    return device

# Only predicts S parameter.
class MLP(nn.Module):
    def __init__(self, input_size, model_order):
        super().__init__()
        # Hecht-Nelson method to determine the node number of the hidden layer: 
        #   node number of hidden layer is (2n+1) when input layer is (n).
        hidden_size = (2*input_size + 1)
        # The output size is 2 times the model order (len of poles) for the residues,
        # and 2 again since each coeff is a complex value (return 0im if real only). 
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

    # Model convention is to output: residue_i real, residue_i imag, pole_i real, pole_i imag
    # This converts that into complex residues followed by the complex poles.
    def forward(self, x):
        output = self.layers(x)
        complex_output = torch.view_as_complex(output.view(-1, 2))
        return complex_output

    def loss_fn(self, actual_S, predicted_S):
        # S is complex, but MAPE isn't. Do average of r and i parts.
        real_MAPE = mean_absolute_percentage_error(actual_S.real, predicted_S.real)
        imag_MAPE = mean_absolute_percentage_error(actual_S.imag, predicted_S.imag)
        return (real_MAPE + imag_MAPE) / 2

