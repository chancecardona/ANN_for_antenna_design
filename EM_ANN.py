# Get current dir
# Load the matlab file

# Get ['candidates'] and ['responses']

# For candidate in candidates:

# For result in results:

# Train each branch

# Only need to train and test the S parameter (S11).

# input: x (vector of geometrical variables) 
# output: TF coefficients of S-parameter (this is what we want to train)

# Loss: H(s) = Sigma(r_i / (s - p_i)) from i=1 to Q (Q is the order of the TF)


## Hard for chaotic TF orders to train ANN accurately. 
## -> group original training samples into different categories C_k (K categories)
## Then lets say order of each category (Q) = k, the index of the category. (group category by TF order)

# Train SVM:
#https://github.com/chancecardona/HCR-Project3/blob/master/fit_svm.py

## SVM Output: Q' = {Q'_1, K, Q'_K1}
## TF order (ground truth): Q = {Q_1, K, Q_K1}
# 

# Classify:

# Train ANN:
# EM simulation results:
## O = {O_1, K, O_W} , where W is the number of sample points of that frequency.

# Outputs of pole-residue-based transfer function:
## O' = {O'_1, K, O'_W}





### Eventually there will be 3 branches:
# Branch 2 uses Gain instead of S-Parameter, Branch 3 uses Radiation Pattern (angle) as input to vector fitting prior to classification.
