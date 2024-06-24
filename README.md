# Multiparameter Modeling with ANN for Antenna Design Example Implementation

This is based on the paper [Multiparameter Modeling with ANN for Antenna Design]('./Multiparameter Modeling With ANN for Antenna Design.pdf')  

## Environment
I tested this with Debian 12 (and Ubuntu 22.04) using Python 3.9.

### Installation
You should also have CUDA installed along with necessary drivers.
```
pip3 install -r requirements.txt
```

## Running
### Training
```
python3 EM_ANN.py --train
```

### Inference
If using Git-LFS you should be able to run in inference mode right away.
```
python3 EM_ANN.py
```

## Info
### Data Format
There are 2 data files, a training one used in the ANN training process, and a testing one used for ANN validation after training.  

Each matlab data file has 2 rows "candidates", and "responses":
- candidates: Geometrical Variables. 
  This should be 3 geometric variables considered during design, such as l_p (patch length), l_a (aperture length), and h_c (cavity height).
- responses: [Freq, Real(S-Param), Imaginary(S-Param)]

## Results:
The results are promising! 
We have a similar error value for the SVM's as the literature.
The ANNs' error value is slightly higher (around double the literature's S-parameter MAPE, but in line with their other parameter MAPEs).
This slight difference in accuracy could be resolved with model fine tuning such as adjusting the hyperparameters more, training each order of ANN more (such as order 10, which had few data), or by splitting each ANN into two (one for poles, one for residue).

#### SVM MAPE 
- Training: 0.09344956259018758%
- Testing SVM MAPE is: 0.08918650793650794%
#### ANN MAPE
- Training Average: 1.1146725757496145%
- Testing Average: 1.1433655905223012%

Overall this shows accurate replication of the paper.
More results from parameter selection are included below.

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
#### ANNs MAPE (results during parameter selection)
- Training Average using 3 total epochs: 1.2962557324024175%
- Testing Average using 3 total epochs: 1.2560414383900285%
- Higher epochs lowered this, likely due to overfitting.

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
For specifics on the TF layout and interaction: [Parametric Modeling of EM Behavior of Microwave Components using Combined NN's and Hybrid TF's](https://www.researchgate.net/publication/340908715_Parametric_Modeling_of_EM_Behavior_of_Microwave_Components_Using_Combined_Neural_Networks_and_Hybrid-Based_Transfer_Functions/fulltext/5ea38b8392851c1a906d0b23/Parametric-Modeling-of-EM-Behavior-of-Microwave-Components-Using-Combined-Neural-Networks-and-Hybrid-Based-Transfer-Functions.pdf)
For the NN implementation defaults such as number of layers please see: "Neural Network Approached to Electromagnetic Based Modeling of Passive Components".

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
