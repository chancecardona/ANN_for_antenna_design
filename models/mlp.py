import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.fourier_features as ff

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

# (Not Implemented) Rational Based Transfer Function:
# NN 3: Predict a_i (numerator coeffs) for Rational Transfer Function
# NN 4: Predict b_i (denominator coeffs) for Rational Transfer Function 

def mlp_layers(input_size : int, hidden_size : int, output_size : int):
    layers = nn.Sequential(
        # Input layer
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        # First hidden layer (Fully Connected)
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        # Second hidden layer (Fully Connected)
        #nn.Linear(hidden_size, hidden_size),
        #nn.ReLU(),
        # Output layer
        nn.Linear(hidden_size, output_size),
    )
    return layers

# Only predicts S parameter.
class MLP(nn.Module):
    def __init__(self, input_size : int, model_order : int):
        super().__init__()
        self.model_order = model_order
        # Define fourier features here since x is low dimensional
        # model order is the number of output features
        fourier_features_size = model_order * 2 # TODO 2 for the residues and the poles, and 2 for the real and imag.
        self.fourier_features = ff.FourierFeatures(input_size, fourier_features_size, scale=100)
        fourier_output_size = 2 * fourier_features_size # 2 for sin and cos.
        # Hecht-Nelson method to determine the node number of the hidden layer: 
        # node number of hidden layer is (2n+1) when input layer is (n).
        #hidden_size = (2*input_size + 1)
        self.hidden_size = (2*fourier_output_size + 1)
        # The output size is 2 times the model order (len of poles) since each coeff is a complex value (return 0im if real only). 
        # +1 because one model predicts d, the other predicts e for the PoleResidueTF.
        self.output_size = (model_order * 2 + 1)# * 2

        # Define the mlp layers
        self.layers = mlp_layers(fourier_output_size, self.hidden_size, self.output_size)

        # Not using double currently as https://discuss.pytorch.org/t/problems-with-target-arrays-of-int-int32-types-in-loss-functions/140/2
        #self.double()

        # Also define the optimizer here so we don't need to keep track elsewhere.
        lr = max(0.09, min(0.2 * (self.model_order / 8)**8, 0.98)) # Want LR of about 0.2 for 8, 0.5 for 10, clamp between .1 to ~1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Reduces lr by a factor of gamma every step_size epochs.
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.85)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Fourier Feature the x data since it's low dim and this normalizes it
        x_fourier = self.fourier_features(x)

        # Get the d, e constants, the poles, and the residues.
        pred_coeffs = self.layers(x_fourier)#[0]
        num_poles = self.model_order * 2 # real and imag
        const_coeff, coeffs = pred_coeffs[0], pred_coeffs[1:].reshape(self.model_order, 2)
        complex_coeffs = torch.complex(coeffs[:, 0], coeffs[:, 1])
        output = torch.cat((const_coeff.unsqueeze(0), complex_coeffs), dim=0)
        return output
