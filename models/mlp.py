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

# Literature (Zhao, Feng, Zhang and Jin's Parametric Modeling of EM Behavior of Microwave... Hybrid-Based Transfer Functions) breaks the NN into 2:
# NN 1: Predict poles (p_i) for H(s) (Pole-Residue basaed transfer function)
# NN 2: Predict residue (r_i) for H(s)
# My approach uses only 1 : NN that predicts residues, then poles, alternating between real and imag. 

# Only predicts S parameter.
class MLP(nn.Module):
    def __init__(self, input_size, model_order):
        super().__init__()
        # Hecht-Nelson method to determine the node number of the hidden layer: 
        # node number of hidden layer is (2n+1) when input layer is (n).
        hidden_size = (2*input_size + 1)
        # The output size is 2 times the model order (len of poles) since each coeff is a complex value (return 0im if real only). 
        output_size = model_order * 2
        self.p_layers = nn.Sequential(
            # Input layer
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # First hidden layer (Fully Connected)
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # Output layer
            nn.Linear(hidden_size, output_size),
        )
        self.r_layers = nn.Sequential(
            # Input layer
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # First hidden layer (Fully Connected)
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # Output layer
            nn.Linear(hidden_size, output_size),
        )
        # If run into performance issues try converting input data instead of model
        self.double()

    # Model convention is to output: residue_i real, residue_i imag, pole_i real, pole_i imag
    # This converts that into complex poles and then residues
    def forward(self, x):
        complex_p = torch.view_as_complex(self.p_layers(x).view(-1, 2))
        complex_r = torch.view_as_complex(self.r_layers(x).view(-1, 2))
        output = torch.cat((complex_p, complex_r), dim=0)
        return output

    def loss_fn(self, actual_S, predicted_S):
        # Take complex conjugate to do square
        c = predicted_S - actual_S
        return torch.abs(torch.sum(c * torch.conj(c)) / 2)

    # Used for model evaluation
    def error_mape(self, actual_S, predicted_S):
        # S is complex, but MAPE isn't. Do average of r and i parts.
        real_MAPE = mean_absolute_percentage_error(actual_S.real, predicted_S.real)
        imag_MAPE = mean_absolute_percentage_error(actual_S.imag, predicted_S.imag)
        return (real_MAPE + imag_MAPE) / 2

