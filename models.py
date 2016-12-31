import numpy as np
import tensorflow as tf


class TwoLayerNet(object):
    """docstring for ClassName"""
    def __init__(self, n_hidden_1=80, n_input= 192, n_classes = 99):
        super(TwoLayerNet, self).__init__()
        self.n_hidden_1 = n_hidden_1
        self.n_classes = n_classes
        self.n_input = n_input
        
        # tf Graph input/output
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        # init ~ 1/sqrt(n) * random sample from normal distribution
        # zero biases
        self.weights = {
            'h1': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(self.n_input)),
                                            tf.random_normal([self.n_input, self.n_hidden_1]))),
            'out': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(self.n_hidden_1)),
                                            tf.random_normal([self.n_hidden_1, self.n_classes])))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden_1])),
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }

        # function that computes output of neural net
        self.predict = self.compute(self.x, self.weights, self.biases)

        # loss function
        # with L2-regularization term
        self.cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict, self.y))
                + 0.001*tf.nn.l2_loss(self.weights['h1'])
                + 0.001*tf.nn.l2_loss(self.weights['out'])
                + 0.001*tf.nn.l2_loss(self.biases['b1'])
                + 0.001*tf.nn.l2_loss(self.biases['out']))


    # Create model computation
    def compute(self,x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        # Output layer with linear activation
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer  




class ThreeLayerNet(object):
    """docstring for ClassName"""
    def __init__(self, n_hidden_1=100, n_hidden_2=50, n_input= 192, n_classes = 99):
        super(ThreeLayerNet, self).__init__()
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_classes = n_classes
        self.n_input = n_input
        

        # tf Graph input
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        # init ~ 1/sqrt(n) * random sample from normal distribution
        # zero biases
        self.weights = {
            'h1': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(self.n_input)),
                                            tf.random_normal([self.n_input, self.n_hidden_1]))),
            'h2': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(self.n_hidden_1)),
                                           tf.random_normal([self.n_hidden_1, self.n_hidden_2]))),
            'out': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(self.n_hidden_2)),
                                            tf.random_normal([self.n_hidden_2, self.n_classes])))
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([self.n_hidden_1])),
            'b2': tf.Variable(tf.zeros([self.n_hidden_2])),
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }


        # function that computes output of neural net
        self.predict = self.compute(self.x, self.weights, self.biases)

        # loss function
        # with L2-regularization term
        self.cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict, self.y))
                + 0.001*tf.nn.l2_loss(self.weights['h1'])
                + 0.001*tf.nn.l2_loss(self.weights['h2'])
                + 0.001*tf.nn.l2_loss(self.weights['out'])
                + 0.001*tf.nn.l2_loss(self.biases['b1'])
                + 0.001*tf.nn.l2_loss(self.biases['b2'])
                + 0.001*tf.nn.l2_loss(self.biases['out']))


    # Create model computation
    def compute(self,x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        #layer_1 = tf.nn.elu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        #layer_2 = tf.nn.elu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer  