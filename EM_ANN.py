import argparse 
import os
import numpy as np
import scipy.io # Read Matlab files
from sklearn import svm # SVM
from sklearn.metrics import mean_absolute_percentage_error, classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as mplt
import torch

from neuro_tf_utils import *

GHz = 1e9

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
    # Y is S_11 over the frequency range (GHz) with 3 vals per sample representing: [frequency (GHz), real, imaginary]
    # W is number of points in freq space
    
    X = training_data["candidates"]
    Y = training_data["responses"]
    X_test = test_data["real_test_candidates"]
    Y_test = test_data["real_test_responses"]
        
    # Allocate relevant items as tensors on the appropriate device (e.g. GPU)
    device = get_device()
    tensor_X = torch.FloatTensor(X).to(device)
    tensor_X_test = torch.FloatTensor(X_test).to(device)
    Y_data = np.array([y[0] for y in Y])
    Y_test_data = np.array([y[0] for y in Y_test])
    freqs = Y_data[:, :, 0]
    S_11_samples_train = Y_data[:, :, 1] + Y_data[:, :, 2] * 1j
    S_11_samples_test = Y_test_data[:, :, 1] + (Y_test_data[:, :, 2] * 1j)
    tensor_freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
    tensor_S_train = torch.tensor(S_11_samples_train, dtype=torch.complex64, device=device)
    tensor_S_test  = torch.tensor(S_11_samples_test, dtype=torch.complex64, device=device)
    
    if args.train: 
        print("Beginning training.")
        # Vector Fitting 
        # Just say the vector fitting results are our "observations" for now.
        # Return a list of vector-fitting objects that have been fit for later use.
        vf_samples = vector_fitting(Y)
        # Classify the testing data here too for later.
        vf_samples_test = vector_fitting(Y_test)

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
        # SVC for versatility in parameters, LinearSVC may be preferrable.
        # tried over ['linear', 'poly', 'rbf', sigmoid']
        svc = svm.SVC(kernel='sigmoid')
        # Scale data with the StandardScaler
        clf = make_pipeline(StandardScaler(), svc)
        clf.fit(X, model_orders_observed)

        print("SVM has been fit, saving to pickle.")
        joblib.dump(clf,"model_weights_output/svm.pkl")
        
        # SVM predict on Train Data for a sanity check. 
        print(f"Train Actual Model Orders (VF): {model_orders_observed}")
        model_orders_predicted = clf.predict(X)
        print(f"Train Predicted Model Orders: {model_orders_predicted}")
        
        # SVM predict on Test Data
        print(f"Test Actual (VF) Model Orders: {model_orders_observed}")
        model_orders_test_predicted = clf.predict(X_test)
        print(f"Test Predicted Model Orders: {model_orders_test_predicted}")
          
        # Evaluate Average training Accuracy
        train_accuracy = accuracy_score(model_orders_observed, model_orders_predicted)
        train_conf_matrix = confusion_matrix(model_orders_observed, model_orders_predicted)
        train_cls_report = classification_report(model_orders_observed, model_orders_predicted, zero_division=0)
        print(f"Training SVM Accuracy is: {train_accuracy}%")
        print(f"Training SVM Confusion Matrix\n", train_conf_matrix)
        print(f"Training SVM Classification Report", train_cls_report)
        
        # Evaluate Average testing Accuracy
        test_accuracy = accuracy_score(model_orders_test_observed, model_orders_test_predicted)
        test_conf_matrix = confusion_matrix(model_orders_test_observed, model_orders_test_predicted)
        test_cls_report = classification_report(model_orders_test_observed, model_orders_test_predicted, zero_division=0)
        print(f"Testing SVM Accuracy is: {test_accuracy}%")
        print(f"Testing SVM Confusion Matrix\n", test_conf_matrix)
        print(f"Testing SVM Classification Report", test_cls_report)
        
        ## Train ANN on EM simulation results and Outputs of pole-residue-based transfer function: ##
        print(f"Training ANNs now...")

        ANNs = create_neural_models(vf_samples, tensor_X, tensor_S_train, tensor_freqs, plot=args.plot)
        print("Pre-training on vector-fitting coefficients finished. Beginning fine-tuning with training data.")
        train_neural_models(ANNs, model_orders_predicted, tensor_X, tensor_S_train, tensor_freqs)

        print("Training finished, saving models.")
        for order,models in ANNs.items():
            torch.save(models[0], f"model_weights_output/s_param_ann_order_{order}_p.pkl")
            torch.save(models[1], f"model_weights_output/s_param_ann_order_{order}_r.pkl")
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
          
        
        ANNs = {}
        for order in set(np.concatenate([model_orders_predicted, model_orders_test_predicted])):
            models = [torch.load(f"model_weights_output/s_param_ann_order_{order}_p.pkl"),
                      torch.load(f"model_weights_output/s_param_ann_order_{order}_r.pkl")]
            ANNs[order] = models

    print(f"Now beginning inference.")
    # Sanity check with Training data
    print("Starting sample run on training (if loss not low, model failed to fit)")
    S_predicted_samples_train, train_loss_avg = predict_samples(ANNs, model_orders_predicted, tensor_X, tensor_S_train, tensor_freqs)
    print("Average training MAPE:", train_loss_avg*100)

    # Test data
    print("Starting test run for actual accuracy.")
    S_predicted_samples_test, test_loss_avg = predict_samples(ANNs, model_orders_test_predicted, tensor_X_test, tensor_S_test, tensor_freqs)
    print("Average testing MAPE:", test_loss_avg*100)
   
    # Plot neural net predictions
    #if args.plot:
    if True:
        print("Plotting Train data")
        for i in range(len(Y)):
            mplt.plot(freqs[i], 20*np.log10(np.abs(S_11_samples_train[i])), 'r-', label="Source (HFSS)")
            mplt.plot(freqs[i], 20*np.log10(np.abs(S_predicted_samples_train[i].detach().numpy())), 'b-.', label="ANN")
            mplt.xlabel("Frequency (GHz)")
            mplt.ylabel("S_11 (dB)")
            mplt.title(f"Order {model_orders_predicted[i]}")
            mplt.show()
        print("Plotting Test data")
        for i in range(len(Y_test)):
            mplt.plot(freqs[i], 20*np.log10(np.abs(S_11_samples_test[i])), 'r-')
            mplt.plot(freqs[i], 20*np.log10(np.abs(S_predicted_samples_test[i].detach().numpy())), 'b-.')
            mplt.xlabel("Frequency (GHz)")
            mplt.ylabel("S_11 (dB)")
            mplt.title(f"Order {model_orders_test_predicted[i]}")
            mplt.show()

    ### Eventually there will be 3 branches:
    # - S Parameter
    # - Gain
    # - Radiation Pattern (angle)
    # will all be input to vector-fitting for classification.
