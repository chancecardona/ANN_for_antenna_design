# Multiparameter Modeling with ANN for Antenna Design Example Implementation

This is based on the paper [Multiparameter Modeling with ANN for Antenna Design]('./Multiparameter Modeling With ANN for Antenna Design.pdf')  

### Assumptions
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
For specifics on the TF and NN structure see: [Parametric Modeling of EM Behavior of Microwave Components using Combined NN's and Hybrid TF's](https://www.researchgate.net/publication/340908715_Parametric_Modeling_of_EM_Behavior_of_Microwave_Components_Using_Combined_Neural_Networks_and_Hybrid-Based_Transfer_Functions/fulltext/5ea38b8392851c1a906d0b23/Parametric-Modeling-of-EM-Behavior-of-Microwave-Components-Using-Combined-Neural-Networks-and-Hybrid-Based-Transfer-Functions.pdf)

We also need to sort the ANN's by the TF order.
So we first need to pre-train the ANN using vector fitting.
Get the TF coefficients, use the len of them for each sample as the "Y" to train a SVM.
Once the SVM is trained we can use it to predict test data, then sort to proper ANN.

Vector Fitting: Obtains the poles and residues of the TF corresponding to the input set of S-params.
   The set of poles/residue coefficients is the Order.
   Typically each (complex) pole corresponds to a resonance peak in the frequency response, so order should at least match that.
   If there are no resonance peaks (very smooth) should check if soln is not well-defined.
   See Gustavsen1999 for more info.

   Vary the # of poles (1 complex is 2 poles) and eval RMS? 
   Also need SVD (of k rows of the a_bar values), then take its S values and make sure none are small (under machine precision limit ~1e-12). 

### Data Format
There are 2 data files, a training one used in the ANN training process, and a testing one used for ANN validation after training.  

Each data file has 2 rows "candidates", and "responses":
- candidates: Geometrical Variables. 
  This should be 3 geometric variables considered during design, such as l_p (patch length), l_a (aperture length), and h_c (cavity height).
- responses: [Freq, Real(S-Param), Imaginary(S-Param)]
