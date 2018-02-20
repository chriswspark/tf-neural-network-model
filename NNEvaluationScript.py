"""
Evaluate prediction model for USDCAD spot rate
"""
__version__ = '0.2'
__author__ = 'Chris Park'

import pandas as pd                                                         # Version 0.22.0
import numpy as np                                                          # Version 1.14.0
from NN import NN                                                           # Version 0.2
from utils import load_obj                                                  # Version 0.1

# Read Inputs ----------------------------------------------------------------------------------------------------------
input_root = 'Inputs/NN'
output_root = 'Outputs/NN'

eval_data = pd.read_csv(input_root+'OoS FX Data.csv', index_col=0)
trained_model = load_obj(output_root+'trained_model')             # Can load saved parameters here to skip training
m = load_obj(output_root+'model_parameters')                      # Can load saved parameters here to skip training

# Parse Inputs ---------------------------------------------------------------------------------------------------------
# Load model parameters
eval_data = eval_data.fillna(method='ffill')                      # Replace missing data with last valid observation

usdcad_spot = eval_data['USDCAD Curncy']
eval_data = eval_data.drop('USDCAD Curncy', axis=1)               # usdcad removed as a feature

x = eval_data.T.as_matrix().astype('float32')
y = usdcad_spot.T.as_matrix().astype('float32')
y = y.reshape((1, len(y)))                                        # reshape y to be a vector
y_labels = ['usdcad_spot']

# Model setup ----------------------------------------------------------------------------------------------------------
num_features = x.shape[0]
num_examples = x.shape[1]
assert (y.shape[1] == x.shape[1])                               # Check that input and outputs have same num_examples

# Initialize Neural Network
fx_prediction_model = NN(alpha=m['alpha'], num_epochs=m['num_epochs'], mini_batch_size=m['mini_batch_size'],
                         adam_beta1=m['adam_beta1'], adam_beta2=m['adam_beta2'], adam_epsilon=m['adam_epsilon'],
                         lambda_reg=m['lambda_reg'], drop_rate=m['drop_rate'], seed=m['seed'],
                         layer_dims=m['layer_dims'], activation_functions=m['activation_functions'])


print('Total data set Model Performance ------------------')
x_normalized = np.log(x)
full_data_metrics, full_data_predictions = fx_prediction_model.evaluate_model(x_normalized, y, trained_model)

# Daily time series over the entire data set of prediction values ------------------------------------------------------
writer = pd.ExcelWriter(output_root+'evaluation_full_data.xlsx')
pd.DataFrame(x.T, index=eval_data.index, columns=eval_data.columns).to_excel(writer, 'inputs')
pd.DataFrame(full_data_predictions, index=eval_data.index, columns=y_labels).to_excel(writer, 'predictions')
pd.DataFrame(y.T, index=eval_data.index, columns=y_labels).to_excel(writer, 'outputs')
writer.close()
