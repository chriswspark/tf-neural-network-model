"""
Evaluate prediction model for USDCAD spot rate (moving up or down or flat)
"""
__version__ = '0.2'
__author__ = 'Chris Park'

import pandas as pd                                                         # Version 0.22.0
import numpy as np                                                          # Version 1.14.0
from SoftmaxNN import SoftmaxNN                                             # Version 0.2
from utils import normalize_inputs, load_obj                                # Version 0.1

# Read Inputs ----------------------------------------------------------------------------------------------------------
input_root = 'Inputs/SoftmaxNN/'
output_root = 'Outputs/SoftmaxNN/'

eval_data = pd.read_csv(input_root+'OoS FX Data.csv', index_col=0)
part_i_predictions = pd.read_excel('Outputs/NN/evaluation_full_data.xlsx', sheet_name='predictions', index_col=0)
trained_model = load_obj(output_root+'trained_model')             # Can load saved parameters here to skip training
m = load_obj(output_root+'model_parameters')                      # Can load saved parameters here to skip training

# Parse Inputs ---------------------------------------------------------------------------------------------------------
eval_data = eval_data.fillna(method='ffill')                       # Replace missing data with last valid observation

eval_data['USDCAD_FV'] = part_i_predictions.squeeze()              # Load part i predictions as a feature
usdcad_spot = eval_data['USDCAD Curncy']
eval_data = eval_data.drop('USDCAD Curncy', axis=1)                # usdcad removed as a feature
x = eval_data.iloc[:-1]                                            # x should only be historical data
x = x.T.as_matrix().astype('float32')

# Set up output layer as a multi-classification output for softmax regression
usdcad_return = usdcad_spot.iloc[1:].as_matrix() - usdcad_spot.iloc[:-1].as_matrix()    # Daily diff for usd/cad
y = np.zeros((3, len(usdcad_return)), dtype=np.float32)
tol = 1e-5
up_days = (usdcad_return < -tol)
down_days = (usdcad_return > tol)
flat_days = ~(up_days | down_days)
y[0, :] = 1.0 * (down_days)
y[1, :] = 1.0 * (flat_days)
y[2, :] = 1.0 * (up_days)

y_labels = ['down', 'flat', 'up']

# Model setup ----------------------------------------------------------------------------------------------------------
num_features = x.shape[0]
num_examples = x.shape[1]
assert (y.shape[1] == x.shape[1])                               # Check that input and outputs have same num_examples

# Initialize Neural Network
fx_prediction_model = SoftmaxNN(alpha=m['alpha'], num_epochs=m['num_epochs'], mini_batch_size=m['mini_batch_size'],
                                adam_beta1=m['adam_beta1'], adam_beta2=m['adam_beta2'], adam_epsilon=m['adam_epsilon'],
                                lambda_reg=m['lambda_reg'], drop_rate=m['drop_rate'], seed=m['seed'],
                                layer_dims=m['layer_dims'], activation_functions=m['activation_functions'])


print('Total data set Model Performance ------------------')
x_normalized = np.log(x)
x_normalized, _, _ = normalize_inputs(x_normalized, m['mu'], m['sigma'])
full_data_metrics, full_data_predictions = fx_prediction_model.evaluate_model(x_normalized, y, trained_model)

# Daily time series over the entire data set of prediction values ------------------------------------------------------
writer = pd.ExcelWriter(output_root+'evaluation_full_data.xlsx')
pd.DataFrame(x.T, index=eval_data.index[:-1], columns=eval_data.columns).to_excel(writer, 'inputs')
pd.DataFrame(full_data_predictions, index=eval_data.index[:-1], columns=y_labels).to_excel(writer, 'predictions')
pd.DataFrame(y.T, index=eval_data.index[:-1], columns=y_labels).to_excel(writer, 'outputs')
writer.close()
