import matplotlib.pyplot as mplt
import numpy as np
import skrf
import torch
from torchmetrics.functional.regression import mean_absolute_percentage_error as torch_mean_absolute_percentage_error
from models.mlp import MLP, get_device


# Y is the frequency response of S_param. Should have n values of shape [freq, real, imag]
def vector_fitting(Y : np.ndarray, verbose : bool = False, plot : bool = False) -> np.ndarray:
    n_samples = len(Y)
    W = len(Y[0][0])
    # For each candidate sample, we have W [freq, r, i] S-param values
    # Reserve a list for each input sample to its vectorfitting object result.
    samples_vf = []
    
    for i in range(n_samples):
        # Get the complex and real components for each freq sample
        S_11 = Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
        freqs = Y[i][0][:, 0]
    
        # Assuming S=s_11
        ntwk = skrf.Network(frequency=freqs, s=S_11, name=f"frequency_response_{i}")
        vf = skrf.VectorFitting(ntwk)
        vf.auto_fit()

        samples_vf.append(vf)
        model_orders = vf.get_model_order(vf.poles)
        if verbose:
            print("Vector Fitting candidate sample", i)
            print(f'model order for sample {i} = {model_orders}')
            print(f'n_poles_real = {np.sum(vf.poles.imag == 0.0)}')
            print(f'n_poles_complex = {np.sum(vf.poles.imag > 0.0)}')
            print(f'RMS Error = {vf.get_rms_error()}')
        if plot: 
            fig, ax = mplt.subplots(2, 1)
            fig.set_size_inches(6, 8)
            vf.plot_convergence(ax=ax[0]) 
            vf.plot_s_db(ax=ax[1])
            mplt.tight_layout()
            mplt.show()
          
    return samples_vf

# Pole Residue Transfer Function:
# H(s) = Sigma(r_i / (s - p_i)) from i=1 to Q (Q is the order of the TF)
# TODO: Need to make sure all poles are smooth/continuous for this one?
# Assumes coefficients are all complex data type.
def PoleResidueTF(d : float, e : float, poles : torch.Tensor, residues : torch.Tensor, freqs : torch.Tensor) -> torch.Tensor:
    epsilon = 1e-9
    device = get_device()
    # H is freq response
    H = torch.zeros(len(freqs), dtype=torch.cdouble).to(device)
    # s is the angular complex frequency
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

# Used for model evaluation
# Using the loss given in eq 10 of (Ding et al.: Neural-Network Approaches to EM-Based Modeling of Passive Components)
def loss_fn(actual_S : torch.Tensor, predicted_S : torch.Tensor) -> float:
    # Take complex conjugate to do square
    c = predicted_S - actual_S
    return torch.sum(torch.abs(c * torch.conj(c))) / 2
    #return torch.sum(torch.abs(torch.pow(c, 2))) / 2

def error_mape(actual_S : torch.Tensor, predicted_S : torch.Tensor) -> float:
    # S is complex, but MAPE isn't. Do average of r and i parts.
    real_MAPE = torch_mean_absolute_percentage_error(actual_S.real, predicted_S.real)
    imag_MAPE = torch_mean_absolute_percentage_error(actual_S.imag, predicted_S.imag)
    return (real_MAPE + imag_MAPE) / 2

def predict(p_model, r_model, input_X : torch.Tensor, freqs : torch.Tensor) -> torch.Tensor: 
    if p_model.training:
        pred_e_poles = p_model.forward(input_X)
        pred_d_residues = r_model.forward(input_X)
        pred_e, pred_poles = pred_e_poles[0], pred_e_poles[1:]
        pred_d, pred_residues = pred_d_residues[0], pred_d_residues[1:]
        #print("ANN Poles", pred_poles.detach().cpu().numpy())
        #print("ANN Residues", pred_residues.detach().cpu().numpy())
        return pred_d, pred_e, pred_poles, pred_residues
    else:
        # Don't calculate gradients in eval mode
        with torch.no_grad():
            pred_e_poles = p_model.forward(input_X)
            pred_d_residues = r_model.forward(input_X)
            pred_e, pred_poles = pred_e_poles[0], pred_e_poles[1:]
            pred_d, pred_residues = pred_d_residues[0], pred_d_residues[1:]
            return pred_d, pred_e, pred_poles, pred_residues


def create_neural_models(vf_series : list, X : torch.Tensor, Y : torch.Tensor, freqs : torch.Tensor, plot : bool = False) -> dict:
    x_dims = len(X[0])
    # This assumes the model order is actually the length of the poles array, and sorts
    # according to that rather than the actual TF order since that corresponds to neuron layers
    # to enable treating real and complex coefficients the same.
    model_orders = [len(vf.poles) for vf in vf_series]
    order_set = set(model_orders)
    ANNs = {}

    device = get_device()
    
    # Create an MLP for each order found
    for order in order_set:
        # Real and Imag vectors the same length, even if term is purely Real, so we return complex from mlp.
        # Form: [const, {poles, residues}] for poles and residues.
        models = [MLP(x_dims, order).to(device), MLP(x_dims, order).to(device)]
        # ANN_k corresponds to the order of the tf, but the nodes in ANN_k correspond to the len of the vectors for the coefficients.
        ANNs[order] = models
 
    # Go through each sample, sort by the order (that we got earlier),
    # predict the coefficients with the ANN's, feed that into the TF, and calc loss with the vector fit coeffs fed through the TF.
    epochs = 3
    for epoch in range(0,epochs):
        print(f"Starting Epoch {epoch}")
        current_loss = 0.0
        for i in range(len(model_orders)):
            model_order = model_orders[i]
            models = ANNs[model_order]
            # Zero out the grad each train step
            [model.optimizer.zero_grad() for model in models]
           
            # Predict S_11 via the vector fit coeffs (all residues first, then all poles)
            # Start with constant coeffs, going to assume only 1 per response function.
            vf_d = vf_series[i].constant_coeff.item()
            vf_e = vf_series[i].proportional_coeff.item()
            vf_poles = torch.from_numpy(vf_series[i].poles).to(device)
            vf_residues = torch.from_numpy(vf_series[i].residues[0]).to(device)
            vf_S = PoleResidueTF(vf_d, vf_e, vf_poles, vf_residues, freqs[i])

            # Predict S_11 via the ANN
            pred_d, pred_e, pred_poles, pred_residues = predict(models[0], models[1], X[i], freqs[i])
            pred_S = PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs[i])
      
            freqs_np = freqs.detach().cpu().numpy()
            if plot:
                S_samples = Y[i].detach().cpu().numpy()
                print(f"SAMPLE {i} of ORDER {model_order}") 
                print(f"VF RMS error {vf_series[i].get_rms_error()}")
                print("VF Consts", vf_d, vf_e)
                print("VF Poles", vf_series[i].poles)
                print("VF Residues", vf_series[i].residues)
                #print("Source", S_samples)
                #print("VF", vf_S)
                #print("ANN", pred_S.detach().cpu().numpy())
                fig, ax = mplt.subplots(2, 1)
                fig.set_size_inches(6, 8)
                #vf.plot_convergence(ax=ax[0]) 
                vf_series[i].plot_s_db(ax=ax[1])
                ax[0].plot(freqs_np[i], 20*np.log10(np.abs(S_samples)), 'r-', label="Source (HFSS)")
                ax[0].plot(freqs_np[i], 20*np.log10(np.abs(vf_S.detach().cpu().numpy())), 'g--', label="Vector Fit")
                ax[0].plot(freqs_np[i], 20*np.log10(np.abs(pred_S.detach().cpu().numpy())), 'b-.', label="Predicted (ANN)")
                ax[0].set_xlabel("Frequency (GHz)")
                ax[0].set_ylabel("S_11 (dB)")
                ax[0].set_ylabel("S_11 (dB)")
                ax[0].set_title(f"Order {model_order}")
                ax[0].legend()
                mplt.tight_layout()
                mplt.show()
            
            #loss = loss_fn(vf_S, pred_S)
            # Calculate Pre trainLoss on just the coefficients (multiplied by a constant for gradient descent)
            loss = 10000 * (torch.norm(vf_d - pred_d, p=2) + \
                   torch.norm(vf_e - pred_e, p=2) + \
                   torch.norm(vf_poles - pred_poles, p=2) + \
                   torch.norm(vf_residues - pred_residues, p=2))
            loss.backward()
            [model.optimizer.step() for model in models]
            current_loss += loss.item()
        
            if  i != 0 and i%10 == 0:
                print(f"Loss after mini-batch (model order {model_order}) %5d: %.3f"%(i, current_loss/500))
                current_loss = 0.0

        # Increment lr scheduler
        for _,models in ANNs.items():
            [model.scheduler.step() for model in models]
    
    # Set models to eval mode now for inference. Set back to train if training more.
    for _,models in ANNs.items():
        [model.eval() for model in models]

    return ANNs

# X is the geometrical input to the model.
# Y is only used for training after the predicted coefficients are plugged in.
def train_neural_models(ANNs : dict, model_orders : np.ndarray, X : torch.Tensor, Y : torch.Tensor, freqs : torch.Tensor):
    device = get_device()
    # Set models to train mode for training in case they're in eval.
    for _,models in ANNs.items():
        [model.train() for model in models]
    # Go through each sample, sort by the order (that we got earlier),
    # predict the coefficients with the ANN's, feed that into the TF, and calc loss with the baseline S-param.
    epochs = 15
    for epoch in range(0,epochs):
        print(f"Starting Epoch {epoch}")
        current_loss = 0.0
        for i in range(len(X)):
            model_order = model_orders[i]
            models = ANNs[model_order]

            # Zero out the grad each train step
            [model.optimizer.zero_grad() for model in models]
            
            # Get ground truth data from Y
            S_11 = Y[i]

            # Predict S_11 via the ANN
            pred_d, pred_e, pred_poles, pred_residues = predict(models[0], models[1], X[i], freqs[i])
            pred_S = PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs[i])
            
            # Calculate Loss
            loss = loss_fn(S_11, pred_S)
            loss.backward()
            [model.optimizer.step() for model in models]
            current_loss += loss.item()
        
            if i != 0 and i%10 == 0:
                print(f"Loss after mini-batch (model order {model_order}) %5d: %.3f"%(i, current_loss/500))
                current_loss = 0.0

        # Increment lr scheduler
        for _,models in ANNs.items():
            [model.scheduler.step() for model in models]

    # Set models to eval mode now for inference.
    for model_order,models in ANNs.items():
        [model.eval() for model in models]

def predict_samples(ANNs : dict, model_orders : np.ndarray, X : torch.Tensor, Y : torch.Tensor, freqs : torch.Tensor) -> tuple[list, float]:
    device = get_device()
    # Filter based on test observation
    # Get order for each sample.
    S_predicted_samples = []
    S_predicted_mape_avg = 0.0
    for i in range(len(model_orders)):
        S_11 =  Y[i]
        model_order = model_orders[i]
        models = ANNs[model_order]

        # Predict S_11
        pred_d, pred_e, pred_poles, pred_residues = predict(models[0], models[1], X[i], freqs[i])
        pred_S = PoleResidueTF(pred_d, pred_e, pred_poles, pred_residues, freqs[i])
        S_predicted_samples.append(pred_S)
    
        # Calculate Loss
        loss = loss_fn(S_11, pred_S)
        S_predicted_mape_avg += error_mape(S_11, pred_S).item()
        if i%10 == 0:
            print(f"Loss of prediction {i}: {loss.item()}")
    S_predicted_mape_avg /= len(model_orders)
    return S_predicted_samples, S_predicted_mape_avg
