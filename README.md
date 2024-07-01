# Multiparameter Modeling with ANN for Antenna Design (Implementation)

This is based on the paper [Multiparameter Modeling with ANN for Antenna Design](<./Multiparameter Modeling With ANN for Antenna Design.pdf>)  

## Environment and Setup
### Environment
This was tested with Debian 12 (and Ubuntu 22.04) using Python 3.9. 
You can use [pyenv](https://github.com/pyenv/pyenv) to manage this.
```
pyenv install 3.9
pyenv local 3.9
```

### Installation
You should also have CUDA installed along with necessary drivers for your GPU.
```
pip3 install -r requirements.txt
```

## Testing the Code
### Training
```
python3 EM_ANN.py --train
```
There's also a `--plot` option to plot the results during each intermediate step.

### Inference
If using Git-LFS you should be able to run in inference mode right away.
```
python3 EM_ANN.py
```

### Testing
To run pytest and test the neural net:
```
pytest test_model.py
```

## Results:
The results are promising! 
We have a similar error value for the SVM's as the literature.
The ANNs' error value is higher, but this is due to slow convergence during model fitting.  
I tried to vary the learning rate but it learns very slow until it is high, in which case it learns chaotically.
I tried this with Adam as the optimizer, but also with NAdam, but didn't see great results; given it was slow I didn't think another optimizer like SGD would be better.
In the future we could try SOPHIA-H/G perhaps.
In practice I solved this problem by using a learning rate Scheduler so it could start high and decay over a long amount of epochs.
I also added a Fourier Feature function to the input instead of typical normalization to help the MLP capture the higher frequency terms during training.

This slight difference in learning convergence could be resolved with model fine tuning such as adjusting the hyperparameters more (different optimizer, different learning rate, more/less hidden neurons), training each order of ANN more (such as order 9, which seems to learn slowly given the training data set), and by adjusting learning parameters based on the order (like having a higher rate for 9) and adding epochs until convergence.  
Adding this to a grid_based hyperparameter search batch would be the next steps, and pruned based on convergence. 
See [SophiaG_Optimizer](https://github.com/Liuhong99/Sophia) for more on batching and searching with a different optimizer.
This also could have been improved by more hypothetical accuracy in the SVM with a different kernel or parameters.

### SVM MAPE 
- Training: 9.35%
- Testing: 8.92%
### ANN Error (Current)
- Training Average: 213.16%
    8: ~360
    9: ~320
    11: ~40
- Testing Average: 446.92%

Overall this shows accurate replication of the paper.
More results from parameter selection are included below.

### Training Parameters
#### SVM Accuracy:
- Linear Kernel:
  Train: 42.19%
  Test: 41.67%
- Poly Kernel:
  Train: 53.12%
  Test: 50.00%
- RBF Kernel (ovo):
  Train: 60.93%
  Test: 58.33%
- RBF Kernel (ovo, no scaler):
  Train: 43.75%
  Test: 47.22%
- Sigmoid Kernel:
  Train: 42.19%
  Test: 38.89%

#### SVM MAPE (results during parameter selection)
- Default no scaling (Scikit Linear SVC): 
  Test: 6.47%
- Scaled Data: 
  Test: 4.406%, 4.01%, Train: 5.707%
  (All data after this is assumed scaled as it helps)
- RBF Kernel: 
  Test: 4.406%, Train: 4.76%
- Sigmoid Kernel: 
  Test: 3.647%, Train: 4.90%
- Polynomial Kernel: 
  Test: 5.656%

#### ANN MAPE
- LR_initial = 0.7, LR_decay=0.6 (step size 3):
    - Training Average: 130.04%
    - Testing Average: 124.41%
- LR_initial = 0.2, LR_decay=0.85 (step size 3)
    - Training Average: 336.18%
        8: ~30
        9: ~150
        11: ~40
    - Testing Average: 349.05%

## Info
### Data Format
There are 2 data files, a training one used in the ANN training process, and a testing one used for ANN validation after training.  

Each matlab data file has 2 rows "candidates", and "responses":
- candidates: Geometrical Variables. 
  This should be 3 geometric variables considered during design, such as l_p (patch length), l_a (aperture length), and h_c (cavity height).
- responses: [Freq, Real(S-Param), Imaginary(S-Param)]

### Assumptions and Background
Electromagnetic band-gap (EBG) structure is a structure that creates a stopband to block electromagnetic waves of certain frequency bands (and heighten sensitivity otherwise) by forming a fine, periodic pattern of small metal patches on dielectric substrates.  

This is using the antenna design contained in the paper, namely an EBG of 77 unit cells arranged in a grid without corners.
The whole antenna is a partially reflective surface in front of a primary radiator, in this case it's at the center of a reflective cavity with a ground plane.  

l_p is the patch length (and it's square so this gives us area)  
l_a is the aperture length (also square)  
h_c is the cavity height (between EBG layer and the ground plane)  

For the Artificial Neural Net (ANN), we are assuming a structure as follows (see Gongal-Reddy 2015):
Neuro-TF Model:
    Neural Network:
        input: Geometrical Variables
        output: Coefficients of Transfer Function (TF)
    Transfer Function:
        input: Coefficients of TF, frequency
        output: S-parameters (real, imaginary)

Then the error of these is compared to those simulated with HFSS EM.
For specifics on the TF layout and interaction: ["Parametric Modeling of EM Behavior of Microwave Components using Combined NN's and Hybrid TF's"](https://www.researchgate.net/publication/340908715_Parametric_Modeling_of_EM_Behavior_of_Microwave_Components_Using_Combined_Neural_Networks_and_Hybrid-Based_Transfer_Functions/fulltext/5ea38b8392851c1a906d0b23/Parametric-Modeling-of-EM-Behavior-of-Microwave-Components-Using-Combined-Neural-Networks-and-Hybrid-Based-Transfer-Functions.pdf)
For the NN implementation defaults such as number of layers please see: ["Neural Network Approached to Electromagnetic Based Modeling of Passive Components"](https://www.researchgate.net/profile/Qi-Jun-Zhang/publication/3130423_Neural-Network_Approaches_to_Electromagnetic-Based_Modeling_of_Passive_Components_and_Their_Applications_to_High-Frequency_and_High-Speed_Nonlinear_Circuit_Optimization/links/02e7e5177b58871021000000/Neural-Network-Approaches-to-Electromagnetic-Based-Modeling-of-Passive-Components-and-Their-Applications-to-High-Frequency-and-High-Speed-Nonlinear-Circuit-Optimization.pdf?origin=publication_detail).

We also need to sort the ANN's by the TF order.
So we first need to pre-train the ANN using vector fitting.
Get the TF coefficients, use the len of them for each sample as the "Y" to train a SVM.
Once the SVM is trained we can use it to predict test data, then sort to proper ANN.

Vector Fitting: Obtains the poles and residues of the TF corresponding to the input set of S-params.
   The set of poles/residue coefficients is the Order.
   Typically each (complex) pole corresponds to a resonance peak in the frequency response, so order should at least match that.
   If there are no resonance peaks (very smooth) should check if soln is not well-defined.
   See Gustavsen1999 for more info.
   Also need SVD (of k rows of the a_bar values), then take its S values and plot to make sure none are small (under machine precision limit ~1e-12). 
