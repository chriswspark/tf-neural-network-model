"""
Neural Network Architecture for Softmax regression.
"""
__version__ = '0.2'
__author__ = 'Chris Park'

import tensorflow as tf                                                     # Version 1.4.1
from NN import NN                                                           # Version 0.2
from forward_propagation import forward_propagation                         # Version 0.1


class SoftmaxNN(NN):
    """
    Neural network for softmax regression.

    Parameters (from NN) -----------------------------------------------------------------------------------------------
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

    Inherited methods (from NN) ----------------------------------------------------------------------------------------
    __init__(self, alpha=0.0001, num_epochs=3000, mini_batch_size=64, layer_dims=None, activation_functions=None,
            adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, lambda_reg=0.0, drop_rate=0.0, seed=None):

    set_layer_dims(self, x, y, num_nodes):

    randomize_mini_batches(self, x, y, mini_batch_size=None):

    initialize_parameters(self, n_x, n_y, layer_dims=None):

    train_model(self, x_train, y_train, layer_dims=None, print_cost=True)
    """

    # Override methods -------------------------------------------------------------------------------------------------
    def set_activation_functions(self, hidden_activation='relu', output_activation='softmax'):
        """
        Set activation functions for each layer in neural network (excl. input layer)
        :param hidden_activation: hidden layer activation function. Default = ReLU
        :param output_activation: output layer activation function. Default = softmax
        """
        activation_functions = [hidden_activation]*(self.num_layers-2)
        activation_functions.insert(0, 'N/A')                                   # no activation for input layer
        activation_functions.extend([output_activation])
        self.activation_functions = activation_functions
        print('activation_functions set to:')
        print(self.activation_functions)

    def compute_cost(self, Z, y, parameters):
        """
        Calculate cross-entropy loss for softmax regression.
        :param Z: output layer linear unit from forward propagation
        :param y: np.array of shape (num_classifiers, num_examples) of true labels
        :return: cost: cost function value
        """
        logits = tf.transpose(Z)
        labels = tf.transpose(y)
        L = len(parameters) // 2                        # number of weights

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        reg_loss = 0                                    # initialize regularization loss
        if self.lambda_reg > 0:
            for l in range(1, L):
                reg_loss += tf.nn.l2_loss(parameters['W'+str(l)])

        cost = tf.reduce_mean(loss + self.lambda_reg*reg_loss)

        return cost

    def evaluate_model(self, x_test, y_test, parameters, print_results=True):
        """
        Evaluate accuracy, F1-score, precision and recall of trained model on given inputs/outputs.
        :param x_test: np.array of shape (num_features, num_examples) of inputs
        :param y_test: np.array of shape (num_classifiers, num_examples) of true labels
        :param parameters: dictionary containing parameters Wl, bl
        :param print_results: if True, print metrics.
        :return: metrics: dictionary of model metrics (Accuracy, F1-Score, Precision, and Recall)
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

            prediction = tf.one_hot(tf.argmax(ZL), n_classifiers)

            correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

            t_p = tf.count_nonzero(tf.argmax(ZL) * tf.argmax(y), dtype=tf.float32)              # true positive
            f_p = tf.count_nonzero(tf.argmax(ZL) * (tf.argmax(y) - 1), dtype=tf.float32)        # false positive
            f_n = tf.count_nonzero((tf.argmax(ZL) - 1) * tf.argmax(y), dtype=tf.float32)        # false negative

            precision = tf.divide(t_p, tf.add(t_p, f_p))
            recall = tf.divide(t_p, tf.add(t_p, f_n))
            f1_score = tf.divide(2 * tf.multiply(precision, recall), tf.add(precision, recall))

            metrics['Accuracy'] = accuracy.eval({x: x_test, y: y_test})
            metrics['F1-Score:'] = f1_score.eval({x: x_test, y: y_test})
            metrics['Precision:'] = precision.eval({x: x_test, y: y_test})
            metrics['Recall:'] = recall.eval({x: x_test, y: y_test})

            if print_results:
                for met_label, met in sorted(metrics.items()):
                    print(met_label, met)

            predictions = prediction.eval({x: x_test})

            return metrics, predictions
