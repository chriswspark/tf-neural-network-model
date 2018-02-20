"""
Forward propagation module for NN.py
"""
__version__ = '0.1'
__author__ = 'Chris Park'

import tensorflow as tf                                                     # Version 1.4.1


def linear_forward(A, W, b):
    """
    Compute linear part of layer's forward propagation.
    :param A: activation from previous layer of shape (size of previous layer, num_examples)
    :param W: weights matrix of shape (size of current layer, size of previous layer)
    :param b: bias vector of shape (size of current layer, 1)
    :return: Z: pre-activation parameter
             # cache: dictionary containing A, W, b - for use in backward_propagation
    """
    Z = tf.add(tf.matmul(W, A),b)

    return Z


def linear_activation_forward(A_prev, W, b, activation, drop_rate, training):
    """
    Compute forward propagation (Linear->Activation) for a single layer.
    :param A_prev: activation from previous layer of shape (size of previous layer, num_examples)
    :param W: weights matrix of shape (size of current layer, size of previous layer)
    :param b: bias vector of shape (size of current layer, 1)
    :param activation: string denoting activation function ('relu', 'sigmoid', 'tanh', 'softmax')
    :param drop_rate: dropout regularization rate. e.g. drop_rate = 0.1 would drop out 10% of units
    :param training: boolean denoting whether in training mode (apply dropout) or inference mode (do not apply dropout)
    :return: A: activation of current layer
    """
    Z = linear_forward(A_prev, W, b)

    if activation == 'sigmoid':
        A = tf.nn.sigmoid(Z)
    elif activation == 'relu':
        A = tf.nn.relu(Z)
    elif activation == 'tanh':
        A = tf.nn.tanh(Z)
    elif activation == 'softmax':
        A = tf.nn.softmax(Z)

    A = tf.layers.dropout(A, rate=drop_rate, training=training)

    return A


def forward_propagation(x, parameters, activation_functions, drop_rate=1.0, training=False):
    """
    Implement forward propagation.
    :param x: data (np.array) of shape (num_features, num_examples)
    :param parameters: output of initialize_parameters
    :param activation_functions: list containing activation function for each layer
    :param drop_rate: dropout regularization rate. e.g. drop_rate = 0.1 would drop out 10% of units
    :return: Z: output layer linear unit
    """
    L = len(parameters) // 2                                                # number of layers
    A = x                                                                   # A0 = x

    for l in range(1, L):
        A_prev = A
        A = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation_functions[l],
                                      drop_rate, training)

    Z = linear_forward(A, parameters['W'+str(L)], parameters['b'+str(L)])

    return Z