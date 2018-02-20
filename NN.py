"""
Neural Network Architecture for regression.
"""
__version__ = '0.2'
__author__ = 'Chris Park'

import numpy as np                                                          # Version 1.14.0
import matplotlib.pyplot as plt                                             # Version 2.1.1
import tensorflow as tf                                                     # Version 1.4.1
from forward_propagation import forward_propagation                         # Version 0.1


class NN(object):
    """
    Neural network for regression.

    Parameters ---------------------------------------------------------------------------------------------------------
    :param alpha: learning rate of model
    :param num_epochs: number of epochs of the optimization loop
    :param mini_batch_size: integer representing size of mini-batches
    :param layer_dims: list containing dimensions of each layer in network
    :param activation_functions: list containing activation function for each layer
    :param adam_beta1: Exponential decay parameter for the past gradients estimates (for Adam optimizer)
    :param adam_beta2: Exponential decay parameter for the past squared gradients estimates (for Adam optimizer)
    :param adam_epsilon: small value to prevent division by zero in Adam steps
    :param lambda_reg: L2-regularization parameter
    :param drop_rate: dropout regularization rate. e.g. drop_rate = 0.1 would drop out 10% of units
    :param seed: fixed seed for randomization
    """
    def __init__(self, alpha=0.0001, num_epochs=3000, mini_batch_size=64, layer_dims=None, activation_functions=None,
                 adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, lambda_reg=0.0, drop_rate=0.0, seed=None):
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.layer_dims = layer_dims
        self.activation_functions = activation_functions
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.lambda_reg = lambda_reg
        self.drop_rate = drop_rate
        self.seed = seed
        if layer_dims is not None:
            self.num_layers = len(layer_dims)
        else:
            self.num_layers = 0

    def set_layer_dims(self, x, y, num_nodes):
        """
        Set dimensions of each layer in network (attribute: layer_dims)
        :param x: np.array of shape (num_features, num_examples) of inputs
        :param y: np.array of shape (num_classifiers, num_examples) of true labels
        :param num_nodes: list of containing number of nodes for each hidden layer (excl. input and output layer)
        """
        layer_dims = num_nodes
        layer_dims.insert(0, x.shape[0])                                        # num input nodes = num_features
        layer_dims.extend([y.shape[0]])                                         # num output nodes = num_classifiers
        self.layer_dims = layer_dims
        self.num_layers = len(self.layer_dims)                                  # update number of layers
        print('layer_dims set to:')
        print(self.layer_dims)
        print('num_layers = ' + str(self.num_layers))

    def set_activation_functions(self, hidden_activation='relu', output_activation='sigmoid'):
        """
        Set activation functions for each layer in neural network (excl. input layer)
        :param hidden_activation: hidden layer activation function. Default = ReLU
        :param output_activation: output layer activation function. Default = Sigmoid
        """
        activation_functions = [hidden_activation]*(self.num_layers-2)
        activation_functions.insert(0, 'N/A')                                   # no activation for input layer
        activation_functions.extend([output_activation])
        self.activation_functions = activation_functions
        print('activation_functions set to:')
        print(self.activation_functions)

    def randomize_mini_batches(self, x, y, mini_batch_size=None):
        """
        Partition x, y into random mini-batches.
        :param x: np.array of shape (num_features, num_examples) of inputs
        :param y: np.array of shape (num_classifiers, num_examples) of true labels
        :param mini_batch_size: integer representing size of mini-batches. If None is passed, use self.mini_batch_size
        :return: mini_batches: list of mini-batches (x_i, y_i)
        """

        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        m = x.shape[1]                                              # number of examples
        num_mini_batches = int(np.floor(m/mini_batch_size))       # number of complete mini-batches

        mini_batches = []

        # Shuffle data prior to splitting
        np.random.seed(self.seed)
        permutation = list(np.random.permutation(m))
        shuffled_x = x[:, permutation]
        shuffled_y = y[:, permutation]

        # Partition into mini-batches
        for i in range(num_mini_batches):
            tmp_mini_batch_x = shuffled_x[:, i*mini_batch_size:(i+1)*mini_batch_size]
            tmp_mini_batch_y = shuffled_y[:, i*mini_batch_size:(i+1)*mini_batch_size]

            tmp_mini_batch = (tmp_mini_batch_x, tmp_mini_batch_y)
            mini_batches.append(tmp_mini_batch)

        # Last mini-batch is smaller than mini_batch_size, so handle separately
        if m % mini_batch_size != 0:
            last_mini_batch_x = shuffled_x[:, :-mini_batch_size*num_mini_batches]
            last_mini_batch_y = shuffled_y[:, :-mini_batch_size*num_mini_batches]

            last_mini_batch = (last_mini_batch_x, last_mini_batch_y)
            mini_batches.append(last_mini_batch)

        return mini_batches

    def initialize_parameters(self, n_x, n_y, layer_dims=None):
        """
        Initialize variables x,y and parameters  for neural network -
        :param n_x: scalar representing number of features in input layer
        :param n_y: scalar representing number of classifiers in output layer
        :param layer_dims: list containing dimensions of each layer in network
        :return: parameters: dictionary containing tensors Wl, bl:
                            Wl: weight matrix of shape (layer_dims[l], layer_dims[l-1])
                            b1: bias vector of shape (layer_dims[l], 1)
                 x: placeholder of shape [n_x, None]
                 y: placeholder of shape [n_y, None]
        """
        parameters = {}                                                     # initialize parameters dictionary

        if layer_dims is None:
            layer_dims = self.layer_dims

        L = len(layer_dims)                                                 # number of layers

        # Create tensors for each weight and bias
        for l in range(1, L):
            parameters['W'+str(l)] = tf.get_variable('W'+str(l), shape=[layer_dims[l], layer_dims[l-1]],
                                                     initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            parameters['b'+str(l)] = tf.get_variable('b'+str(l), shape=[layer_dims[l], 1],
                                                     initializer=tf.zeros_initializer())

        # Create placeholders for x, y
        x = tf.placeholder(tf.float32, shape=(n_x, None))
        y = tf.placeholder(tf.float32, shape=(n_y, None))

        return x, y, parameters

    def compute_cost(self, Z, y, parameters):
        """
        Calculate mean squared error cost for regression.
        :param Z: output layer linear unit from forward propagation
        :param y: np.array of shape (num_classifiers, num_examples) of true labels
        :return: cost: cost function value
        """
        L = len(parameters) // 2                            # number of weights

        loss = tf.reduce_mean(tf.squared_difference(Z, y))

        reg_loss = 0                                        # initialize regularization loss
        if self.lambda_reg > 0:
            for l in range(1, L):
                reg_loss += tf.nn.l2_loss(parameters['W'+str(l)])

        cost = tf.reduce_mean(loss + self.lambda_reg*reg_loss)

        return cost

    def evaluate_model(self, x_test, y_test, parameters, print_results=True):
        """
        Evaluate prediction error (mse) of trained model on given inputs/outputs.
        :param x_test: np.array of shape (num_features, num_examples) of inputs
        :param y_test: np.array of shape (num_classifiers, num_examples) of true labels
        :param parameters: dictionary containing parameters Wl, bl
        :param print_results: if True, print metrics.
        :return: metrics: dictionary of model metrics (mse)
                 predictions: array of predicted outputs
        """
        metrics = {}

        (n_features, n_examples) = x_test.shape
        n_classifiers = y_test.shape[0]

        with tf.Session():
            # Create placeholders for x, y
            x = tf.placeholder(tf.float32, shape=(n_features, None))
            y = tf.placeholder(tf.float32, shape=(n_classifiers, None))

            ZL = forward_propagation(x, parameters, self.activation_functions, drop_rate=0.0, training=False)

            prediction = ZL

            mse = tf.reduce_mean(tf.squared_difference(ZL, y))                  # evaluate model on mean squared error
            tss = tf.reduce_mean(tf.squared_difference(y, tf.reduce_mean(y)))   # total squared sum
            rsq = tf.subtract(1.0, tf.divide(mse, tss))                         # R^2

            metrics['MSE:'] = mse.eval({x: x_test, y: y_test})
            metrics['R^2:'] = rsq.eval({x: x_test, y: y_test})

            if print_results:
                for met_label, met in sorted(metrics.items()):
                    print(met_label, met)

            predictions = prediction.eval({x: x_test}).T

            return metrics, predictions

    def train_model(self, x_train, y_train, layer_dims=None, print_cost=True):
        """
        Train a L-layer neural network for regression.
        :param x_train: np.array of shape (num_features, num_examples) of training inputs
        :param y_train: np.array of shape (num_classifiers, num_examples) of training true labels
        :param layer_dims: list containing dimensions of each layer in network
        :param print_cost: if True, print and plot the cost
        :return: parameters: dictionary containing parameters Wl, bl
                 metrics: dictionary of model metrics (mse)
                 predictions: array of predicted outputs
        """

        (n_features, n_examples) = x_train.shape
        n_classifiers = y_train.shape[0]

        costs = []                                                  # cost over each iteration

        if layer_dims is None:                                      # option to specify different layer_dims
            layer_dims = self.layer_dims

        x, y, parameters = self.initialize_parameters(n_features, n_classifiers, layer_dims)

        # Initialize optimizer
        ZL = forward_propagation(x, parameters, self.activation_functions, self.drop_rate, training=True)
        cost = self.compute_cost(ZL, y, parameters)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha, beta1=self.adam_beta1, beta2=self.adam_beta2,
                                           epsilon=self.adam_epsilon).minimize(cost)

        init = tf.global_variables_initializer()                    # Initialize graph global variables

        # Start session to compute tensorflow graph
        with tf.Session() as sess:

            # Run initialization
            sess.run(init)

            for epoch in range(self.num_epochs):
                epoch_cost = 0.0
                num_mini_batches = int(n_examples/self.mini_batch_size)
                mini_batches = self.randomize_mini_batches(x_train, y_train)

                for tmp_mini_batch in mini_batches:

                    (tmp_X, tmp_Y) = tmp_mini_batch

                    _, mini_batch_cost = sess.run([optimizer, cost], feed_dict={x: tmp_X, y: tmp_Y})

                    epoch_cost += mini_batch_cost / num_mini_batches

                # Print cost every epoch
                if print_cost and epoch % 100 == 0:
                    print('Cost after epoch %i: %f' % (epoch, epoch_cost))
                if print_cost and epoch % 100 == 0:
                    costs.append(epoch_cost)

            # Plot cost curve
            if print_cost:
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('epochs (per 10s)')
                plt.show()

            parameters = sess.run(parameters)

            # Evaluate model on training set and print results
            print('Training set Model Performance ------------------')
            metrics, predictions = self.evaluate_model(x_train, y_train, parameters, print_results=True)

            return parameters, metrics, predictions
