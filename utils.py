"""
Helper functions for PredictionScript to save/load parameters, and split and normalize input/output data prior to
training model.
"""
__version__ = '0.1'
__author__ = 'Chris Park'

import numpy as np                                                          # Version 1.14.0
import pickle


def save_obj(obj, output_path_name):
    with open(output_path_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(output_path_name):
    with open(output_path_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def split_data(x, y, split_perc, shuffle=False, seed=None):
    """
    Split data according to train/cv/test set portions passed in split_perc (rounded to nearest integer index).
    (Optional): shuffle data prior to splitting.
    :param x: np.array of shape (num_features, num_examples) of inputs
    :param y: np.array of shape (num_classifiers, num_examples) of true labels
    :param split_perc: list containing desired train/cv/test set split
    :param shuffle: if True, shuffle data prior to splitting into train/dev/test set
    :return: x_train: training set input of shape (num_features, train set size)
             x_cv: dev/cross-validation set input of shape (num_features, dev set size)
             x_test: test set input of shape (num_features, test set size)
             y_train: vector of training set true labels
             y_cv: vector of dev/cross-validation set true labels
             y_test: vector of test set true labels
    """

    num_features = x.shape[0]                           # Number of features
    num_examples = x.shape[1]                           # Number of training examples
    num_classifications = y.shape[0]                    # Number of classifications
    assert (y.shape[1] == x.shape[1])                   # Check dimensions of y

    # Shuffle data prior to splitting
    if shuffle:
        np.random.seed(seed)
        permutation = list(np.random.permutation(num_examples))
        x = x[:, permutation]
        y = y[:, permutation]

    # Normalize split_perc
    split_perc = (1.0*np.array(split_perc))             # First convert to np.array of floats
    split_perc = split_perc/np.sum(split_perc)          # Convert to percentages

    n_train = int(num_examples*split_perc[0])           # Number of examples in training set (rounded)
    n_cv = int(num_examples*split_perc[1])              # Number of examples in dev/cross-validation set (rounded)
    n_test = num_examples - (n_train + n_cv)

    # Split data set
    x_train = x[:, :n_train]
    y_train = y[:, :n_train]
    x_cv = x[:, n_train:n_train+n_cv]
    y_cv = y[:, n_train:n_train+n_cv]
    x_test = x[:, n_train+n_cv:]
    y_test = y[:, n_train+n_cv:]

    # Check dimensions
    assert (x_train.shape == (num_features, n_train))
    assert (y_train.shape == (num_classifications, n_train))
    assert (x_cv.shape == (num_features, n_cv))
    assert (y_cv.shape == (num_classifications, n_cv))
    assert (x_test.shape == (num_features, n_test))
    assert (y_test.shape == (num_classifications, n_test))

    return x_train, x_cv, x_test, y_train, y_cv, y_test


def normalize_inputs(x, mu=None, sigma=None):
    """
    Apply zero mean, unit variance scaling to inputs x. (Optional) normalize instead by given mu and sigma
    :param x: np.array of shape (num_features, num_examples) of inputs
    :param mu: (optional) vector of means of shape (num_features, 1)
    :param sigma: (optional) vector of standard deviations of shape (num_features, 1)
    :return: norm_x: normalized data of shape (num_features, num_examples)
             mu: vector of means used to normalize x
             sigma: vector of standard deviations used to normalize x
    """

    num_features = x.shape[0]                   # Number of variables

    if mu is None:
        mu = np.mean(x, axis=1).reshape((num_features, 1))
    if sigma is None:
        sigma = np.std(x, axis=1).reshape((num_features, 1))

    # Check that mu and sigma are row vectors with same number of features
    assert (mu.shape == (num_features,1))
    assert (sigma.shape == (num_features, 1))

    norm_x = (x - mu)/sigma                     # Normalize x

    assert (norm_x.shape == x.shape)            # Check that normalized inputs have same dimensions as inputs

    return norm_x, mu, sigma
