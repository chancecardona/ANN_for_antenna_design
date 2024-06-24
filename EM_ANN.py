import argparse 
import os
import numpy as np
import scipy.io # Read Matlab files
from sklearn import svm # SVM
from sklearn.metrics import mean_absolute_percentage_error # Using SKlearn's MAPE for np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import skrf
import matplotlib.pyplot as mplt
import torch

from models.mlp import get_device, MLP

# Y is the frequency response of S_param. Should have n values of shape [freq, real, imag]
def vector_fitting(Y : np.ndarray, verbose : bool = True, plot : bool = False) -> np.ndarray:
    n_samples = len(Y)
    W = len(Y[0][0])
    # For each candidate sample, we have W [freq, r, i] S-param values
    # Reserve a list for each input sample to its vectorfitting object result.
    samples_vf = []
    
    for i in range(n_samples):
        print("Starting candidate sample", i)
        # Get the complex and real components for each freq sample
        S_11 = Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
        #S_dB = 20*log10(S_11)
        freqs = Y[i][0][:, 0]
    
        # TODO: is this fine as the S matrix, just S=s_11?
        ntwk = skrf.Network(frequency=freqs, s=S_11, name=f"frequency_response_{i}")
        vf = skrf.VectorFitting(ntwk)
        vf.auto_fit()

        samples_vf.append(vf)
        model_orders = vf.get_model_order(vf.poles)
        if verbose:
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
def PoleResidueTF(coefficients : torch.Tensor, freqs : np.ndarray) -> np.ndarray:
    epsilon = 1e-9
    if len(coefficients) % 2 != 0: 
        print("Coefficients was not even in length meaning poles and residues dif length.")
        exit(1)
    num_poles = len(coefficients) // 2
    # H is freq response
    H = torch.zeros(len(freqs), dtype=torch.cdouble)
    # s is the frequency
    for s in range(len(freqs)):
        for i in range(num_poles):
            # Avoid numerical overflow or division by 0
            denominator = (freqs[s] - coefficients[i+num_poles])
            if abs(denominator) < epsilon:
                denominator += epsilon * (1 if denominator.real >= 0 else -1)
            H[s] += (coefficients[i] / denominator)
    return H

# (Not Implemented) Rational Based Transfer Function:
# NN 3: Predict a_i (numerator coeffs) for Rational Transfer Function
# NN 4: Predict b_i (denominator coeffs) for Rational Transfer Function 

# Only need Y for the freqs
def create_neural_models(vf_series : list, tensor_X : torch.Tensor, Y : np.ndarray, plot : bool = False) -> dict:
    x_dims = len(tensor_X[0])
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
    epochs = 1
    for epoch in range(0,epochs):
        print(f"Starting Epoch {epoch}")
        current_loss = 0.0
        freqs = Y[0][0][:, 0] # TODO Only need Y because of the freqs here
        for i in range(len(tensor_X)):
            model_order = model_orders[i]
            model = ANNs[model_order]
            optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
           
            # Predict S_11 via the vector fit coeffs (all residues first, then all poles)
            vf_coeffs = torch.from_numpy(np.concatenate((vf_series[i].residues[0], vf_series[i].poles))).to(device)
            vf_S = PoleResidueTF(vf_coeffs, freqs)

            # Predict S_11 via the ANN
            pred_tf_coeffs = model(tensor_X[i])
            pred_S = PoleResidueTF(pred_tf_coeffs, freqs)
       
            if plot:
                S_samples = Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
                print("Candidates:")
                print(vf_coeffs)
                print(pred_tf_coeffs)
                print("Source", S_samples)
                print("VF", vf_S)
                print("ANN", pred_S.detach().numpy())
                mplt.plot(freqs, 20*np.log10(abs(S_samples)), 'r-', label="Source (HFSS)")
                mplt.plot(freqs, 20*np.log10(abs(vf_S)), 'g--', label="Vector Fit")
                mplt.plot(freqs, 20*np.log10(abs(pred_S.detach().numpy())), 'b-.', label="Predicted (ANN)")
                mplt.xlabel("Frequency (GHz)")
                mplt.ylabel("S_11 (dB)")
                mplt.legend()
                mplt.show()
            
            # Calculate Loss
            loss = model.loss_fn(vf_S, pred_S)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
        
            if i%10 == 0:
                print(f"Loss after mini-batch for model order {model_order} %5d: %.3f"%(i+1, current_loss/500))
                current_loss = 0.0

    return ANNs

# X is the geometrical input to the model.
# Y is only used for training after the predicted coefficients are plugged in.
def train_neural_models(ANNs : dict, model_orders : np.ndarray, tensor_X : torch.Tensor, Y : np.ndarray):
    device = get_device()
    # Go through each sample, sort by the order (that we got earlier),
    # predict the coefficients with the ANN's, feed that into the TF, and calc loss with the baseline S-param.
    epochs = 1
    for epoch in range(0,epochs):
        current_loss = 0.0
        for i in range(len(tensor_X)):
            freqs = Y[i][0][:, 0]
            S_11 = torch.from_numpy(Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)).to(device)
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
                print(f"Loss after mini-batch for model order {model_order} %5d: %.3f"%(i+1, current_loss/500))
                current_loss = 0.0

    # Set models to eval mode now for inference. Set back to train if training more.
    for model_order,model in ANNs.items():
        model.eval() 

def predict_samples(ANNs : dict, model_orders : np.ndarray, tensor_X : torch.Tensor, Y : np.ndarray) -> tuple[list, float]:
    device = get_device()
    # Filter based on test observation
    # Get order for each sample.
    S_predicted_samples = []
    S_predicted_loss_avg = 0.0
    for i in range(len(model_orders)):
        freqs = Y[i][0][:, 0]
        S_11 = torch.from_numpy(Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)).to(device)
        model_order = model_orders[i]
        model = ANNs[model_order]

        # Predict S_11
        pred_tf_coeffs = model(tensor_X[i])
        pred_S = PoleResidueTF(pred_tf_coeffs, freqs)
        S_predicted_samples.append(pred_S)
    
        # Calculate Loss
        loss = model.loss_fn(S_11, pred_S)
        S_predicted_loss_avg += loss.item()
        if i%10 == 0:
            print(f"Loss of prediction {i}: {loss.item()}")
    S_predicted_loss_avg /= len(model_orders)
    return S_predicted_samples, S_predicted_loss_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains or evaluates an ANN to predict the S_11 freq response using .mat files")
    parser.add_argument("--train", action="store_true", help="Train ANNs from the matlab data files. (else will load files if present).")
    parser.add_argument("--plot", action="store_true", help="Plot data samples with matplotlib.")
    args = parser.parse_args()

    # Get current dir
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # Load the matlab files
    training_data_path = os.path.join(cur_dir, "data/Training_Data.mat")
    test_data_path = os.path.join(cur_dir, "data/Real_Test_Data.mat")
    
    training_data = scipy.io.loadmat(training_data_path)
    test_data = scipy.io.loadmat(test_data_path)
    
    # X = [lp @ ln @ hc]^T (meters), of shape (64, 3) here.
    # Y is S_11 (dB) over the frequency range (GHz) with 3 vals per sample representing: [frequency (GHz), real, imaginary]
    # W is number of points in freq space
    
    X = training_data["candidates"]
    Y = training_data["responses"]
    X_test = test_data["real_test_candidates"]
    Y_test = test_data["real_test_responses"]
        
    # Allocate relevant items as tensors on the appropriate device (e.g. GPU)
    device = get_device()
    tensor_X = torch.tensor(X, device=device)
    tensor_X_test = torch.tensor(X, device=device)
    
    if args.train: 
        print("Beginning training.")
        # Vector Fitting 
        # Just say the vector fitting results are our "observations" for now.
        # Return a list of vector-fitting objects that have been fit for later use.
        vf_samples = vector_fitting(Y)#, plot=args.plot)
        # Classify the testing data here too for later.
        vf_samples_test = vector_fitting(Y_test)#, plot=args.plot)

        #print("Vector fitting finished, saving to file")
        #for vf in vf_samples:
        #    # Saves to Network name by default
        #    vf.write_npz("model_output_weights/vector_fit_train")
        #for vf in vf_samples_test:
        #    vf.write_npz("model_output_weights/vector_fit_test")

        # vf.get_model_order returns the 'true' order for the freq response, but 
        # we use the length of the poles array instead so we can treat real and complex
        # poles the same.
        model_orders_observed = [len(vf.poles) for vf in vf_samples] 
        model_orders_test_observed = [len(vf.poles) for vf in vf_samples_test]
         
        print("Training SVM now.")
        # Train SVM:
        # Need to predict the Order based on the input S-parameter (over frequency space).
        # SVM Input: geometrical variables
        # SVM Output: TF Order (vector fitting on S-param in f-space)
        # SVM Error: Predicted TF Order - Vector Fit TF Order  
        # SVC for versatility in parameters, LinearSVC may be preferrable.
        # TODO: one versus one, vs one versus rest (ovo vs ovho)
        # ['linear', 'poly', 'rbf', sigmoid']
        svc = svm.SVC(kernel='sigmoid')
        # Scale data with the StandardScaler
        clf = make_pipeline(StandardScaler(), svc)
        clf.fit(X, model_orders_observed)

        print("SVM has been fit, saving to pickle.")
        joblib.dump(clf,"model_weights_output/svm.pkl")
        
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
        
        ## Train ANN on EM simulation results and Outputs of pole-residue-based transfer function: ##
        print(f"Training ANNs now...")

        ANNs = create_neural_models(vf_samples, tensor_X, Y, plot=args.plot)
        print("Pre-training on vector-fitting coefficients finished. Beginning fine-tuning with training data.")
        train_neural_models(ANNs, model_orders_predicted, tensor_X, Y)

        print("Training finished, saving models.")
        for order,model in ANNs.items():
            torch.save(model, f"model_weights_output/s_param_ann_order_{order}.pkl")   
    else: 
        # Else load pre trained models.
        print("Initializing testing environment. Loading weights files.")

        #vf_samples = [vector_fitting()] * len(Y)
        #vf_samples_test = [vector_fitting()] * len(Y_test)
        #print("Loading vector-fit files")
        #file_name = "coefficients_Frequency_Response.npz"
        #for vf in vf_samples:
        #    vf.read_npz(f"model_output_weights/vector_fit_train/{file_name}")
        #for vf in vf_samples_test:
        #    vf.read_npz(f"model_output_weights/vector_fit_test/{file_name}")
        #model_orders_observed = [len(vf.poles) for vf in vf_samples] 
        #model_orders_test_observed = [len(vf.poles) for vf in vf_samples_test]

        print("Loading SVM file pickle.")
        clf = joblib.load("model_weights_output/svm.pkl")     
       
        # SVM predict on Train and Test data
        model_orders_predicted = clf.predict(X)
        model_orders_test_predicted = clf.predict(X_test)
        print(f"Train Predicted: {model_orders_predicted}") 
        print(f"Test Predicted: {model_orders_test_predicted}")
          
        # Evaluate Average training and testing MAPE # TODO: Should this be Chi Squared?
        #err = mean_absolute_percentage_error(model_orders_observed, model_orders_predicted)
        #err = mean_absolute_percentage_error(model_orders_test_observed, model_orders_test_predicted)
        #print(f"Training SVM MAPE is: {err}%") 
        #print(f"Testing SVM MAPE is: {err}%") 
        
        ANNs = {}
        for order in set(np.concatenate([model_orders_predicted, model_orders_test_predicted])):
            model = torch.load(f"model_weights_output/s_param_ann_order_{order}.pkl")    
            ANNs[order] = model 

    print(f"Now beginning inference.")
    # Sanity check with Training data
    S_predicted_samples_train, train_loss_avg = predict_samples(ANNs, model_orders_predicted, tensor_X, Y)
    print("Average training MAPE:", train_loss_avg)

    # Test data
    S_predicted_samples_test, test_loss_avg = predict_samples(ANNs, model_orders_test_predicted, tensor_X_test, Y_test)
    print("Average training MAPE:", test_loss_avg)
   
    # Plot neural net predictions
    if args.plot:
        freqs = Y[0][0][:, 0]
        print("Plotting Train Data")
        for i in range(len(Y)):
            S_samples_train = Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
            mplt.plot(freqs, 20*np.log10(abs(S_samples_train)), 'r-')
            mplt.plot(freqs, 20*np.log10(abs(S_predicted_samples_train[i].detach().numpy())), 'b-.')
            mplt.show()
        print("Plotting Test data")
        for i in range(len(Y_test)):
            S_samples_test = Y_test[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
            mplt.plot(freqs, 20*np.log10(abs(S_samples_test)), 'r-')
            mplt.plot(freqs, 20*np.log10(abs(S_predicted_samples_test[i].detach().numpy())), 'b-.')
            mplt.show()

    ### Eventually there will be 3 branches:
    # - S Parameter
    # - Gain
    # - Radiation Pattern (angle)
    # will all be input to vector-fitting for classification.
