# Multiparameter Modeling with ANN for Antenna Design (Implementation)

This is based on the paper [Multiparameter Modeling with ANN for Antenna Design](<./Multiparameter Modeling With ANN for Antenna Design.pdf>)  

## Environment and Setup
### Environment
This was tested with Debian 12 (and Ubuntu 22.04) using Python 3.9. 
You can use [pyenv](https://github.com/pyenv/pyenv) to manage this.
```
pyenv install 3.9
pyenv local 3.9
sudo apt install git-lfs
```

### Installation
First install the python dependencies.
```
pip3 install -r requirements.txt
git lfs install
```

You should also have CUDA installed along with necessary drivers for your GPU.

Also make sure that you `git lfs pull` if it was not installed prior to pull (or else you will need to train from scratch,  which is ok).

## Testing the Code
### Training
```
python3 EM_ANN.py --train
```
- There's also a `--plot` option to plot the results during each intermediate step.
- There's also a `--finetune` option to load the previously trained and saved model, and train more on it.

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

## Methods:

I attempted a few different NN architectures during training, including adding [Fourier Features](https://arxiv.org/pdf/2006.10739) to extract higher frequency information.

I originally attempted to create 1 Neural Net that would output all pole residue coefficients, but due to them mapping potentially unrelated nonlinear functions I chose to brea them into 2, and have each one predict one of the constant coefficients of the vector fitting function.

I also varied between 1 and 2 hidden layers to see which would fit better, and varied the number of hidden nodes.

I also sampled different loss functions to see if it was feasible to pre train the model to predict the coefficients themselves (and compare with an L2 norm distance), but this didn't result in the best loss function for convergence during training, and was found would make the training part take longer.

## Results:
The results are promising! 
The ANNs' error value for me is higher than the literature, but this is due to slow convergence during model fitting, and me not using more than 18 training epochs for this typically.
Currently the ANN predicts roughly the average function found in each order, and seems to have some trouble predicting the finer patterns found in geometry.
I think this is largely due to just the general shape of the loss function and the optimizer, which seems to get stuck in local minima.
As a result of this, during testing I added another hidden layer, which did result in the function fitting faster, but overall having similar characteristics as the 1 hidden layer version. Further increasing the hidden nodes in this configuration followed this trend, but some model orders like 7 did fit better.

When I used a singleh hidden layer but doubled the hidden nodes count, which may be justified if the poles / residues are discontinuous and thus more nonlinear, the model seemed to fit similarly to the 2 hidden layer version using the same amount of hidden nodes.

Fourier Features greatly improved the training of the SVM order classification, and were used for that step.
For the ANNs however, adding Fourier Features instead of typical normalization from the data didn't seem to have much results in the fit function, even with a large "scale" (the standard deviation of the frequency of sin/cos waves), this typically resulted in the model taking longer to train and fitting roughly the same, but with some more noise.

I ad hoc tried Adam, AdamW, NAdam, and SGD with Momentum as various optimizers. Overall SGD performs the best once the model begins to converge, but NAdam and AdamW tend to combine the best results without running into NaN's during training if a hyperparameter is slightly off, and with the help of a learning rate scheduler can be scaled to more epochs, while having similar testing results as training, indiciating generalization.

### Future
The quickest addition would be to train each model until their loss dropped below a threshold (< ~1.0) and then stop training automatically, as the fewer seen samples (7, 11) are typically underfit with the current scheme unless overfitting the others.
The next step would be to smooth any discontinous poles / residues from the Vector Function, which would reduce the hidden neurons required.
Adding this to a grid_based hyperparameter search batch would be the next steps, and pruned based on convergence. 
See [SophiaG_Optimizer](https://github.com/Liuhong99/Sophia) for more on batching and searching with different optimizers.

This also could have been improved by more hypothetical accuracy in the SVM (perhaps with a custom kernel function for this domain), as the current one tests at about 50% accuracy, and misplacing a response into the wrong order will result in the order needing to retrain for that domain, often incorrectly. 


### Training Results
#### ANN Accuracy:
    Average Train MAPE: 435%
    Average Test: 371% (due to it being more 8 samples)
### SVM Accuracy:
    Average Train: 85%
    Average Test: 50%


### Training Parameter selection Results
#### SVM Parameter selection Accuracy (before "classes equal" which lowered it, but was helped by FourierFeature analysis):
- Linear Kernel:
  Train: 42.19%
  Test: 41.67%
- Poly Kernel:
  Train: 53.12%
  Test: 50.00%
- RBF Kernel:
  Train: 60.93%
  Test: 58.33%
- RBF Kernel (no scaler):
  Train: 43.75%
  Test: 47.22%
- Sigmoid Kernel:
  Train: 42.19%
  Test: 38.89%

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
