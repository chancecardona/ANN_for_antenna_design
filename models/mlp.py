import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.regression import mean_absolute_percentage_error
import numpy as np

def get_device():
    device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
    )
    return device

# Pole Residue Transfer Function:
# H(s) = Sigma(r_i / (s - p_i)) from i=1 to Q (Q is the order of the TF)
# TODO: Need to make sure all poles are smooth/continuous for this one?
# Assumes coefficients are all complex data type.
def PoleResidueTF(d : float, e : float, 
                  poles : torch.Tensor, residues : torch.Tensor, freqs : np.ndarray) -> np.ndarray:
    epsilon = 1e-9
    device = get_device()
    # H is freq response
    H = torch.zeros(len(freqs), dtype=torch.cdouble).to(device)
    # s is the angular complex frequency
    s = torch.from_numpy(2j*np.pi * freqs).to(device)
    for j in range(len(s)):
        #import pdb; pdb.set_trace()
        H[j] += d + s[j]*e
        for i in range(len(poles)):
            p = poles[i]
            r = residues[i]
            #if torch.abs(denominator) < epsilon:
            #   denominator += epsilon * (1 if denominator.real >= 0 else -1)
            if torch.imag(p) == 0:
                H[j] += r / (s[j] - p)
            else:
                H[j] += r / (s[j] - p) + torch.conj(r) / (s[j] - torch.conj(p))
    return H

# Literature (Zhao, Feng, Zhang and Jin's Parametric Modeling of EM Behavior of Microwave... Hybrid-Based Transfer Functions) breaks the NN into 2:
# NN 1: Predict poles (p_i) for H(s) (Pole-Residue basaed transfer function)
# NN 2: Predict residue (r_i) for H(s)
# My approach uses only 1 : NN that predicts residues, then poles, alternating between real and imag. 

# (Not Implemented) Rational Based Transfer Function:
# NN 3: Predict a_i (numerator coeffs) for Rational Transfer Function
# NN 4: Predict b_i (denominator coeffs) for Rational Transfer Function 


# Only predicts S parameter.
class MLP(nn.Module):
    def __init__(self, input_size, model_order):
        super().__init__()
        self.model_order = model_order
        # Hecht-Nelson method to determine the node number of the hidden layer: 
        # node number of hidden layer is (2n+1) when input layer is (n).
        hidden_size = (2*input_size + 1)
        # The output size is 2 times the model order (len of poles) since each coeff is a complex value (return 0im if real only). 
        # +1 because one model predicts d, the other predicts e for the PoleResidueTF.
        output_size = model_order * 2 + 1

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

        # Also define the optimizer here so we don't need to keep track elsewhere.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        # Normalize input data
        x_norm = F.normalize(x, dim=0)
        # Get poles and d const
        poles_d = self.p_layers(x_norm)
        d, x_poles = poles_d[-1], poles_d[:-1]
        # Get residues and e const
        residues_e = self.r_layers(x_norm)
        e, x_residues = residues_e[-1], residues_e[:-1]
        # Layer convention is to output: residue_i real, residue_i imag, pole_i real, pole_i imag
        # This converts that into complex poles and then residues
        complex_p = torch.view_as_complex(x_poles.view(-1, 2))
        complex_r = torch.view_as_complex(x_residues.view(-1, 2))
        output = torch.cat((d.unsqueeze(0), e.unsqueeze(0), complex_p, complex_r), dim=0)
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

    def predict(self, input_X : torch.Tensor, freqs : np.ndarray) -> torch.Tensor: 
        if self.training:
            pred_coeffs = self.forward(input_X)
            pred_d = pred_coeffs[0]
            pred_e = pred_coeffs[1]
            pred_poles = pred_coeffs[2:self.model_order]
            pred_residues = pred_coeffs[self.model_order:]
            #print("ANN Poles", pred_poles.detach().numpy())
            #print("ANN Residues", pred_residues.detach().numpy())
            return PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs)
        else:
            # Don't calculate gradients in eval mode
            with torch.no_grad():
                pred_coeffs = self.forward(input_X)
                pred_d = pred_coeffs[0]
                pred_e = pred_coeffs[1]
                pred_poles = pred_coeffs[2:self.model_order]
                pred_residues = pred_coeffs[self.model_order:]
                return PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs)

