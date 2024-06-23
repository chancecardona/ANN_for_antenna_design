import os
import numpy as np
import scipy.io # Read Matlab files
from sklearn import svm # SVM
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import skrf
import matplotlib.pyplot as mplt

from models.mlp import *

# Y is the frequency response of S_param. Should have n values of shape [freq, real, imag]
def vector_fitting(Y : np.ndarray, verbose : bool = True, display : bool = False) -> np.ndarray:
    n_samples = len(Y)
    W = len(Y[0][0])
    # For each candidate sample, we have W [freq, r, i] S-param values
    #print(f"Sanity check, using {n_samples} samples.")
    # Reserve matrix for the model orders for SVM training
    # Reserve a dict for each input sample to its vectorfitting object result.
    samples_vf = []
    
    for i in range(n_samples):
        print("Starting candidate sample", i)
        # Get the complex and real components for each freq sample
        S_11 = Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
        print("Sanity check, len of S vector:", len(S_11))
        #S_dB = 20*log10(S_11)
        freqs = Y[i][0][:, 0]
    
        # TODO: is this fine as the S matrix, just S=s_11?
        ntwk = skrf.Network(frequency=freqs, s=S_11, name="Frequency Response")
        vf = skrf.VectorFitting(ntwk)
        vf.auto_fit()

        samples_vf.append(vf)
        model_orders = vf.get_model_order(vf.poles)
        if verbose:
            print(f'model order for sample {i} = {model_orders}')
            print(f'n_poles_real = {np.sum(vf.poles.imag == 0.0)}')
            print(f'n_poles_complex = {np.sum(vf.poles.imag > 0.0)}')
            print(f'RMS Error = {vf.get_rms_error()}')
        if display: 
            fig, ax = mplt.subplots(2, 1)
            fig.set_size_inches(6, 8)
            vf.plot_convergence(ax=ax[0]) 
            vf.plot_s_db(ax=ax[1])
            mplt.tight_layout()
            mplt.show()
          
    return samples_vf

# Assumes coefficients are all complex data type.
def PoleResidueTF(coefficients : np.ndarray, freqs : np.ndarray) -> np.ndarray:
    epsilon = 1e-9
    if len(coefficients) % 2 != 0: 
        print("Coefficients was not even in length meaning poles and residues dif length.")
        exit(1)
    num_poles = len(coefficients) // 2
    # H is freq response
    H = np.zeros(len(freqs))
    # s is the frequency
    import pdb; pdb.set_trace()
    for s in range(len(freqs)):
        for i in range(num_poles):
            denominator = (freqs[s] - coefficients[i+num_poles])
            if abs(denominator) < epsilon:
                denominator += epsilon
            H[s] += (coefficients[i] / denominator)
    return H

# Only need Y for the freqs
def create_neural_models(vf_series : list, tensor_X : np.ndarray, Y : np.ndarray) -> dict:
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
        # Real and Imag vectors the same length, even if term is purely Real.
        model = MLP(x_dims, order).to(device)
        # ANN_k corresponds to the order of the tf, but the nodes in ANN_k correspond to the len of the vectors for the coefficients.
        ANNs[order] = model
 
    # Go through each sample, sort by the order (that we got earlier),
    # predict the coefficients with the ANN's, feed that into the TF, and calc loss with the vector fit coeffs fed through the TF.
    for i in range(len(X)):
        freqs = Y[i][0][:, 0] # TODO Only need Y for the freqs here
        # Ordering of coeffs is: (real residues, 
        model_order = model_orders[i]
        model = ANNs[model_order]
        optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
       
        # Predict S_11 via the vector fit coeffs (all residues first, then all poles)
        vf_coeffs = np.concatenate((vf_series[i].residues[0], vf_series[i].poles))
        vf_S = PoleResidueTF(vf_coeffs, freqs)

        # Predict S_11 via the ANN
        pred_tf_coeffs = model(tensor_X[i])
        pred_S = PoleResidueTF(pred_tf_coeffs, freqs)
        
        # Calculate Loss
        loss = model.loss_fn(vf_S, pred_S)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
    
        if i%10 == 0:
            print(f"Loss after mini-batch %5d: %.3f"%(i+1, current_loss/500))
            current_loss = 0.0

    return ANNs

# X is the geometrical input to the model.
# Y is only used for training after the predicted coefficients are plugged in.
def train_neural_models(ANNs : dict, model_orders : np.ndarray, tensor_X : np.ndarray, Y : np.ndarray) -> dict:
    # Go through each sample, sort by the order (that we got earlier),
    # predict the coefficients with the ANN's, feed that into the TF, and calc loss with the baseline S-param.
    for i in range(len(model_orders)):
        freqs = Y[i][0][:, 0]
        S_11 = Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
        model_order = model_orders[i]
        model = ANNs[model_order]
        optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

        # Predict S_11 via the ANN
        pred_tf_coeffs = model(tensor_X[i])
        pred_S = PoleResidueTF(pred_tf_coeffs, freqs)
        
        # Calculate Loss
        loss = model.loss_fn(S_11, pred_S)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
    
        if i%10 == 0:
            print(f"Loss after mini-batch %5d: %.3f"%(i+1, current_loss/500))
            current_loss = 0.0

def predict_samples(ANNs : dict, model_orders : np.ndarray, tensor_X : np.ndarray, Y : np.ndarray):
    # Filter based on test observation
    # Get order for each sample.
    for i in range(len(model_orders)):
        freqs = Y[i][0][:, 0]
        S_11 = Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
        model_order = model_orders[i]
        model = ANNs[model_order]
        optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

        # Predict S_11
        pred_tf_coeffs = model(tensor_X[i])
        pred_S = PoleResidueTF(pred_tf_coeffs, freqs)
    
        # Calculate Loss
        loss = model.loss_fn(S_11, pred_S)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()


if __name__ == "__main__":
    # Get current dir
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Load the matlab files
    training_data_path = os.path.join(cur_dir, "Training_Data.mat")
    test_data_path = os.path.join(cur_dir, "Real_Test_Data.mat")
    
    training_data = scipy.io.loadmat(training_data_path)
    test_data = scipy.io.loadmat(test_data_path)
    
    # X = [lp @ ln @ hc]^T (meters)
    # X is of shape (64, 3)
    # Y is S_11 (dB) over the frequency range (GHz) with 3 vals per sample representing: [frequency (GHz), real, imaginary]
    # W is number of points in freq space
    # K is the total number of categories (number of orders of TF's)
    
    X = training_data["candidates"]
    Y = training_data["responses"]
    X_test = test_data["real_test_candidates"]
    Y_test = test_data["real_test_responses"]
    
    # Vector Fitting 
    # Just say the vector fitting results are our "observations" for now.
    # Return a list of vector-fitting objects that have been fit for later use.
    vf_samples = vector_fitting(Y)
    # vf.get_model_order returns the 'true' order for the freq response, but 
    # we use the length of the poles array instead so we can treat real and complex
    # poles the same.
    model_orders_observed = [len(vf.poles) for vf in vf_samples]
    
    # Train SVM:
    # ['linear', 'poly', 'rbf', sigmoid']
    # Need to predict the Order based on the input S-parameter (over frequency space).
    # SVM Input: geometrical variables
    # SVM Output: TF Order (vector fitting on S-param in f-space)
    # SVM Error: Predicted TF Order - Vector Fit TF Order 
    
    print("Training SVM now.")
    # SVC for versatility in parameters, LinearSVC may be preferrable.
    # TODO: one versus one, vs one versus rest (ovo vs ovho)
    # Scale data with the StandardScaler
    svc = svm.SVC(kernel='sigmoid')
    clf = make_pipeline(StandardScaler(), svc)
    clf.fit(X, model_orders_observed)
    
    
    # Classify:
    vf_samples_test = vector_fitting(Y_test)
    model_orders_test_observed = [len(vf.poles) for vf in vf_samples_test]
    
    # SVM predict on Train Data for a sanity check. 
    model_orders_predicted = clf.predict(X)
    print(f"Train Predicted: {model_orders_predicted}")
    
    # SVM predict on Test Data
    model_orders_test_predicted = clf.predict(X_test)
    print(f"Test Predicted: {model_orders_test_predicted}")
      
    # Evaluate Average training MAPE # TODO: Should this be Chi Squared?
    err = mean_absolute_percentage_error(model_orders_observed, model_orders_predicted)
    print(f"Training SVM MAPE is: {err}%")
    
    # Evaluate Average testing MAPE
    err = mean_absolute_percentage_error(model_orders_test_observed, model_orders_test_predicted)
    print(f"Testing SVM MAPE is: {err}%")
    
    # Train ANN:
    # EM simulation results:
    ## O = {O_1, K, O_W} , where W is the number of sample points of that frequency.
    
    # Outputs of pole-residue-based transfer function:
    ## O' = {O'_1, K, O'_W}
    
    print(f"Training ANNs now...")

    device = get_device()
    # Allocate relevant items as tensors on the appropriate device (e.g. GPU)
    tensor_X = torch.tensor(X, device=device)

    ANNs, model_orders = create_neural_models(vf_samples, tensor_X, Y)
    print(f"Pretraining on vector-fit coefficients finished, beginning training data finetuning.")
    train_neural_models(ANNs, model_orders_predicted, tensor_X, Y)
    
    print(f"Training completed, beginning predictions.")
    tensor_X_test = torch.tensor(X, device=device)
    predict_samples(ANNs, model_orders_test_predicted, tensor_X_test, Y_test)
    
    print("Predictions done, saving model.")
    for order,model in ANNs.items():
        torch.save(model, f"s_param_ann_order_{order}.pkl")
    
    # Only need to train and test the S parameter (S11).
    ## Hecht-Nelson method to determine the node number of the hidden layer: 
    ##   node number of hidden layer is (2n+1) when input layer is (n).
    
    # input: x (vector of geometrical variables) 
    # output: TF coefficients of S-parameter (this is what we want to train)
    
    # Loss: h(PoleResidueTF(x, freq) - S_response)
    
    # Pole Residue Transfer Function:
    # H(s) = Sigma(r_i / (s - p_i)) from i=1 to Q (Q is the order of the TF)
    # Need to make sure all poles are smooth/continuous for this one.
    # NN 1: Predict poles (p_i) for H(s) (Pole-Residue basaed transfer function)
    # NN 2: Predict residue (r_i) for H(s)
    
    # Rational Based Transfer Function:
    # NN 3: Predict a_i (numerator coeffs) for Rational Transfer Function
    # NN 4: Predict b_i (denominator coeffs) for Rational Transfer Function
    
    
    
    
    ### Eventually there will be 3 branches:
    # Branch 2 uses Gain instead of S-Parameter, Branch 3 uses Radiation Pattern (angle) as input to vector fitting prior to classification.
