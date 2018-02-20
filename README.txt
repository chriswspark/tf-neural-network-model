This program is used to train a Neural Network for regression (including Softmax).
Package requirements: tensorflow, pandas, numpy, pickle.

Below are the contents of the program:

NNPredictionScript.py:
    Sample script to parse inputs, set up neural network for regression, train model and save output.

SoftmaxNNPredictionScript.py:
    Sample script to parse inputs, set up neural network for softmax regression, train model and save output.

NN.py:
    Neural Network Architecture for regression. Used in NNPredictionScript.py

SoftmaxNN.py:
    Subclass of NN.py. Neural Network Architecture for Softmax regression. For use in SoftmaxNNPredictionScript.py

forward_propagation.py:
    Forward propagation module for NN.py

utils.py:
    Helper functions for Prediction Script to save/load parameters, and split and normalize input/output data prior to training model.

Inputs/NN/ and Inputs/SoftmaxNN -- inputs for both scripts
../FX Data_table.csv -- USDCAD spot rate data as well as other economic/financial data from Bloomberg.
../OoS FX Data.csv -- Out of sample test data (same format as FX Data_table.csv)

Outputs/NN and Outputs/SoftmaxNN -- outputs for both scripts (identical structure)
../predictions_full_data.xlsx -- predictions, inputs, labels - over the entire data set 
../predictions_by_set.xlsx -- predictions for train/cross-validation/test set as well as corresponding true labels
../stored_parameters.pkl -- set of trained parameters for neural network used to generate predictions. Load using load_obj function
