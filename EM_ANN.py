import os
import numpy as np
import scipy.io # Read Matlab files
from sklearn import svm # SVM
from sklearn.metrics import mean_absolute_percentage_error
import skrf
import matplotlib.pyplot as mplt

# Y is the frequency response of S_param. Should have n values of shape [freq, real, imag]
def vector_fitting(Y : np.ndarray) -> np.ndarray:
    n_samples = len(Y)
    W = len(Y[0][0])
    # For each candidate sample, we have W [freq, r, i] S-param values
    print(f"Sanity check, using {n_samples} samples.")
    # Reserve matrix for the model orders for SVM training
    model_orders = np.zeros(n_samples, dtype=np.uint8)
    
    for i in range(n_samples):
        print("Starting candidate sample", i)
        # Get the complex and real components for each freq sample
        S_11 = Y[i][0][:, 1] + (Y[i][0][:, 2] * 1j)
        print("Sanity check, len of S vector:", len(S_11))
        #S_dB = 20*log10(S_11)
        freqs = Y[i][0][:, 0]
    
        # TODO: is this fine as the S matrix?
        ntwk = skrf.Network(frequency=freqs, s=S_11, name="Frequency Response")
        vf = skrf.VectorFitting(ntwk)
        vf.auto_fit()
        vf.plot_convergence()
       
        model_orders[i] = vf.get_model_order(vf.poles)
        print(f'model order for sample {i} = {model_orders[i]}')
        print(f'n_poles_real = {np.sum(vf.poles.imag == 0.0)}')
        print(f'n_poles_complex = {np.sum(vf.poles.imag > 0.0)}')
        print(f'RMS Error = {vf.get_rms_error()}')
        
        vf.plot_s_db()
          
    return model_orders

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

GHz = 10^9
n_test_samples = len(Y_test)
W_test = len(Y_test[0][0])

# Vector Fitting
model_orders = vector_fitting(Y)

## Hard for chaotic TF orders to train ANN accurately. 
## -> group original training samples into different categories C_k (K categories)
## Then lets say order of each category (Q) = k, the index of the category. (group category by TF order)


# Train SVM:

# Need to predict the Order based on the input S-parameter (over frequency space).
# SVC for versatility in parameters, LinearSVC may be preferrable.
print("Training SVM now.")
clf = svm.SVC()
clf.fit(X, model_orders)
## SVM Training 
# SVM Input: geometrical variables
# SVM Output: TF Order (vector fitting on S-param in f-space)
# SVM Error: Predicted TF Order - Vector Fit TF Order

 

# Classify:
model_orders_test_observed = vector_fitting(Y_test)

# SVM predict on Test Data
model_orders_test_predicted = clf.predict(X_test)
print(f"Predicted: {model_orders_test_predicted}")
  
# Evaluate Average training MAPE
err = mean_absolute_percentage_error(model_orders_test_observed, model_orders_test_predicted)
print(f"Error is: {err}%")

# Train ANN:
# EM simulation results:
## O = {O_1, K, O_W} , where W is the number of sample points of that frequency.

# Outputs of pole-residue-based transfer function:
## O' = {O'_1, K, O'_W}

# Train each branch

# Only need to train and test the S parameter (S11).
## Hecht-Nelson method to determine the node number of the hidden layer: 
##   node number of hidden layer is (2n+1) when input layer is (n).

# input: x (vector of geometrical variables) 
# output: TF coefficients of S-parameter (this is what we want to train)

# Loss: PoleResidueTF(x, freq) - EMResponse(x, freq)

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
