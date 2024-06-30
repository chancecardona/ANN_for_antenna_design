import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.regression import mean_absolute_percentage_error
import numpy as np
import models.fourier_features as ff

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
def PoleResidueTF(d : float, e : float, poles : torch.Tensor, residues : torch.Tensor, freqs : torch.Tensor) -> np.ndarray:
    epsilon = 1e-9
    device = get_device()
    # H is freq response
    H = torch.zeros(len(freqs), dtype=torch.cdouble).to(device)
    # s is the angular complex frequency
    #s = torch.from_numpy(2j*np.pi * freqs).to(device)
    s = 2j*np.pi * freqs
    for j in range(len(s)):
        #import pdb; pdb.set_trace()
        H[j] += d + s[j]*e
        for i in range(len(poles)):
            p = poles[i]
            r = residues[i]
            denominator = (s[j] - p)
            if torch.abs(denominator) < epsilon:
               print("Warning, pole inf detected due to pole singularity.")
               denominator += epsilon * (1 if denominator.real >= 0 else -1)
            if torch.imag(p) == 0:
                H[j] += r / denominator
            else:
                H[j] += r / denominator + torch.conj(r) / (s[j] - torch.conj(p))
    return H

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
        #self.p_layers = mlp_layers(fourier_output_size, hidden_size, output_size)
        #self.r_layers = mlp_layers(fourier_output_size, hidden_size, output_size)
        #self.p_layers = mlp_layers(input_size, hidden_size, output_size)
        #self.r_layers = mlp_layers(input_size, hidden_size, output_size)

        # If run into performance issues try converting input data instead of model
        # Not using double currently as https://discuss.pytorch.org/t/problems-with-target-arrays-of-int-int32-types-in-loss-functions/140/2
        #self.double()

        # Also define the optimizer here so we don't need to keep track elsewhere.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.6)
        # Reduces lr by a factor of gamma every step_size epochs.
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.6)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Normalize input data
        #x_norm = F.normalize(x, dim=0)
        # Fourier Feature the x data since it's low dim
        x_fourier = self.fourier_features(x)

        # Get the d, e constants, the poles, and the residues.
        pred_coeffs = self.layers(x_fourier)
        num_poles = self.model_order * 2 # real and imag
        #d, e, poles, residues = pred_coeffs[0], pred_coeffs[1], pred_coeffs[2:num_poles+2], pred_coeffs[num_poles+2:]
        const_coeff, coeffs = pred_coeffs[0], pred_coeffs[1:].reshape(self.model_order, 2)
        #import pdb; pdb.set_trace()
        #TODO import pdb; pdb.set_trace()
        # Get poles and d const
        #poles_d = self.p_layers(x_fourier)
        #d, x_poles = poles_d[-1], poles_d[:-1]
        # Get residues and e const
        #residues_e = self.r_layers(x_fourier)
        #e, x_residues = residues_e[-1], residues_e[:-1]

        # Layer convention is to output: d, e (reals), complex pole, comples residue (real, imag)
        # This converts that into complex poles and then residues
        #complex_p = torch.view_as_complex(poles.view(-1, 2))
        #complex_r = torch.view_as_complex(residues.view(-1, 2))
        #output = torch.cat((d.unsqueeze(0), e.unsqueeze(0), complex_p, complex_r), dim=0)
        complex_coeffs = torch.complex(coeffs[:, 0], coeffs[:, 1])
        output = torch.cat((const_coeff.unsqueeze(0), complex_coeffs), dim=0)
        return output

# Used for model evaluation
# Using the loss given in eq 10 of (Ding et al.: Neural-Network Approaches to EM-Based Modeling of Passive Components)
def loss_fn(actual_S : torch.Tensor, predicted_S : torch.Tensor) -> float:
    # Take complex conjugate to do square
    c = predicted_S - actual_S
    return torch.sum(torch.abs(c * torch.conj(c))) / 2
    #return torch.sum(torch.abs(torch.pow(c, 2))) / 2

def error_mape(actual_S : torch.Tensor, predicted_S : torch.Tensor) -> float:
    # S is complex, but MAPE isn't. Do average of r and i parts.
    real_MAPE = mean_absolute_percentage_error(actual_S.real, predicted_S.real)
    imag_MAPE = mean_absolute_percentage_error(actual_S.imag, predicted_S.imag)
    return (real_MAPE + imag_MAPE) / 2

def predict(p_model, r_model, input_X : torch.Tensor, freqs : torch.Tensor) -> torch.Tensor: 
    if p_model.training:
        pred_e_poles = p_model.forward(input_X)
        pred_d_residues = r_model.forward(input_X)
        pred_e, pred_poles = pred_e_poles[0], pred_e_poles[1:]
        pred_d, pred_residues = pred_d_residues[0], pred_d_residues[1:]
        #print("ANN Poles", pred_poles.detach().numpy())
        #print("ANN Residues", pred_residues.detach().numpy())
        #return PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs)
        return pred_d, pred_e, pred_poles, pred_residues
    else:
        # Don't calculate gradients in eval mode
        with torch.no_grad():
            pred_e_poles = p_model.forward(input_X)
            pred_d_residues = r_model.forward(input_X)
            pred_e, pred_poles = pred_e_poles[0], pred_e_poles[1:]
            pred_d, pred_residues = pred_d_residues[0], pred_d_residues[1:]
            #return PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs)
            return pred_d, pred_e, pred_poles, pred_residues

