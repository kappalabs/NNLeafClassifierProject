import numpy as np
import tensorflow as tf


class FourierNet(object):

    def __init__(self, filters = 256, n_hidden1=400, stride=1, n_input1=64,n_channels=1,n_input2=128, n_classes = 99):
        super(FourierNet, self).__init__()
        self.n_hidden1 = n_hidden1
        self.n_classes = n_classes
        self.n_input1 = n_input1
        self.filters = filters
        self.stride = stride
        self.n_input2 = n_input2
        self.n_channels = n_channels
        
        # tf Graph input/output
        self.x1 = tf.placeholder("float", [None, n_input1*n_channels])
        self.x2 = tf.placeholder("float", [None, n_input2])
        self.y = tf.placeholder("float", [None, n_classes])
        # Store layers weight & bias
        # init ~ 1/sqrt(n) * random sample from normal distribution
        # zero biases
        self.weights = {
            'c1': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(4*n_channels)),
                                            tf.random_normal([4,n_channels, self.filters]))),
            'h1': tf.Variable(tf.scalar_mul(tf.sqrt(1./float((n_input1//stride)*self.filters+n_input2)),
                                            tf.random_normal([(n_input1//stride)*self.filters+n_input2, self.n_hidden1]))),

            'out': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(self.n_hidden1)),
                                            tf.random_normal([self.n_hidden1, self.n_classes])))
        }
        self.biases = {
            'bc1': tf.Variable(tf.zeros([self.filters])),
            'b1': tf.Variable(tf.zeros([self.n_hidden1])),
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }

        # function that computes output of neural net
        self.predict = self.compute(self.x1, self.x2, self.weights, self.biases)

        # loss function
        # with L2-regularization term
        self.cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict, self.y))
                + 0.0001*tf.nn.l2_loss(self.weights['h1'])
                + 0.0001*tf.nn.l2_loss(self.weights['c1'])
                + 0.0001*tf.nn.l2_loss(self.weights['out'])
                + 0.0001*tf.nn.l2_loss(self.biases['b1'])
                + 0.0001*tf.nn.l2_loss(self.biases['bc1'])
                + 0.0001*tf.nn.l2_loss(self.biases['out']))


    def conv1D(x, W, b, stride=1):
        # Conv1D wrapper, with bias and relu activation
        x = tf.nn.conv1d(x, W, stride, padding="VALID")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # Create model computation function
    def compute(self,x1,x2, weights, biases):

        x1 = tf.reshape(x1, shape=[-1, self.n_input1, self.n_channels])
        act1 = tf.nn.conv1d(x1,weights['c1'],self.stride,padding="SAME")
        act1 = tf.nn.bias_add(act1, biases['bc1'])
        act1 = tf.nn.relu(act1)
        #flatten
        act1= tf.reshape(act1, shape = [-1,(self.n_input1//self.stride)*self.filters])
        act1= tf.concat(1,[act1,x2])

        act2= tf.add(tf.matmul(act1, weights['h1']), biases['b1'])
        act2 =tf.nn.relu(act2)
        # Output layer with linear activation
        out = tf.add(tf.matmul(act2, weights['out']),biases['out'])
        return out  


class ConvNet1D2(object):

    def __init__(self, filters =128,filters2=32,n_hidden=64,stride1=2, stride2=2,n_input=64,n_classes = 99):
        super(ConvNet1D2, self).__init__()
        self.n_classes = n_classes
        self.n_input = n_input
        self.filters = filters
        self.filters2 = filters2
        self.n_hidden = n_hidden
        self.stride1 = stride1
        self.stride2 = stride2

        # tf Graph input/output
        self.x = tf.placeholder("float", [None, n_input*3])
        self.y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        # init ~ 1/sqrt(n) * random sample from normal distribution
        # zero biases
        self.weights = {
            'c1': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(4*3)),
                                            tf.random_normal([4,3, self.filters]))),
    
            'c2': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(4*self.filters)),
                                            tf.random_normal([4,self.filters, self.filters2]))),

            'h1': tf.Variable(tf.scalar_mul(tf.sqrt(1./float((n_input//(stride1*stride2))*self.filters2)),
                                            tf.random_normal(
                                            [(n_input//(stride1*stride2))*self.filters2, self.n_hidden]))),
            'out': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(self.n_hidden)),
                                            tf.random_normal([self.n_hidden, self.n_classes])))
        }
        self.biases = {
            'bc1': tf.Variable(tf.zeros([self.filters])),
            'bc2': tf.Variable(tf.zeros([self.filters2])),
            'b1': tf.Variable(tf.zeros([self.n_hidden])),
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }

        # function that computes output of neural net
        self.predict = self.compute(self.x, self.weights, self.biases)

        # loss function
        # with L2-regularization term
        self.cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict, self.y))
                + 0.0001*tf.nn.l2_loss(self.weights['c1'])
                + 0.0001*tf.nn.l2_loss(self.weights['c2'])
                + 0.0001*tf.nn.l2_loss(self.weights['h1'])
                + 0.0001*tf.nn.l2_loss(self.weights['out'])
                + 0.0001*tf.nn.l2_loss(self.biases['bc1'])
                + 0.0001*tf.nn.l2_loss(self.biases['bc2'])
                + 0.0001*tf.nn.l2_loss(self.biases['b1'])
                + 0.0001*tf.nn.l2_loss(self.biases['out']))


    # for some reason this does not work
    def conv1D(x, W, b, stride=1):
        # Conv1D wrapper, with bias and relu activation
        x = tf.nn.conv1d(x, W, stride, padding="VALID")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # Create model computation function
    def compute(self,x, weights, biases):

        x = tf.reshape(x, shape=[-1, self.n_input, 3])
        #act1= self.conv1D(x,weights['c1'], biases['bc1'])
        act1 = tf.nn.conv1d(x,weights['c1'],self.stride1,padding="SAME")
        act1 = tf.nn.bias_add(act1, biases['bc1'])
        act1 = tf.nn.relu(act1)

        # try dropout
        #act1 = tf.nn.dropout(act1,0.2)

        act2 = tf.nn.conv1d(act1,weights['c2'],self.stride2,padding="SAME")
        act2 = tf.nn.bias_add(act2, biases['bc2'])
        act2 = tf.nn.relu(act2)

        #act2 = tf.nn.dropout(act2,0.2)

        #flatten
        act2= tf.reshape(act2, shape = [-1,(self.n_input//(self.stride1*self.stride2))*self.filters2])

        act3= tf.add(tf.matmul(act2, weights['h1']), biases['b1'])
        act3 =tf.nn.relu(act3)

        # Output layer with linear activation
        out = tf.add(tf.matmul(act3, weights['out']), biases['out'])
        return out  


class ConvNet1D(object):

    def __init__(self, filters = 256, n_hidden_1=400, stride=1, n_input=64, n_classes = 99):
        super(ConvNet1D, self).__init__()
        self.n_hidden_1 = n_hidden_1
        self.n_classes = n_classes
        self.n_input = n_input
        self.filters = filters
        self.stride = stride
        
        # tf Graph input/output
        self.x = tf.placeholder("float", [None, n_input*3])
        self.y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        # init ~ 1/sqrt(n) * random sample from normal distribution
        # zero biases
        self.weights = {
            'c1': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(3*3)),
                                            tf.random_normal([4,3, self.filters]))),
            'h1': tf.Variable(tf.scalar_mul(tf.sqrt(1./float((self.n_input//stride)*self.filters)),
                                            tf.random_normal([(self.n_input//stride)*self.filters, self.n_hidden_1]))),

            'out': tf.Variable(tf.scalar_mul(tf.sqrt(1./float(self.n_hidden_1)),
                                            tf.random_normal([self.n_hidden_1, self.n_classes])))
        }
        self.biases = {
            'bc1': tf.Variable(tf.zeros([self.filters])),
            'b1': tf.Variable(tf.zeros([self.n_hidden_1])),
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }

        # function that computes output of neural net
        self.predict = self.compute(self.x, self.weights, self.biases)

        # loss function
        # with L2-regularization term
        self.cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predict, self.y))
                + 0.0001*tf.nn.l2_loss(self.weights['h1'])
                + 0.0001*tf.nn.l2_loss(self.weights['c1'])
                + 0.0001*tf.nn.l2_loss(self.weights['out'])
                + 0.0001*tf.nn.l2_loss(self.biases['b1'])
                + 0.0001*tf.nn.l2_loss(self.biases['bc1'])
                + 0.0001*tf.nn.l2_loss(self.biases['out']))


    def conv1D(x, W, b, stride=1):
        # Conv1D wrapper, with bias and relu activation
        x = tf.nn.conv1d(x, W, stride, padding="VALID")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # Create model computation function
    def compute(self,x, weights, biases):

        x = tf.reshape(x, shape=[-1, self.n_input, 3])
        #act1= self.conv1D(x,weights['c1'], biases['bc1'])
        act1 = tf.nn.conv1d(x,weights['c1'],self.stride,padding="SAME")
        act1 = tf.nn.bias_add(act1, biases['bc1'])
        act1 = tf.nn.relu(act1)
        #flatten
        act1= tf.reshape(act1, shape = [-1,(self.n_input//self.stride)*self.filters])

        act2= tf.add(tf.matmul(act1, weights['h1']), biases['b1'])
        act2 =tf.nn.relu(act2)
        # Output layer with linear activation
        out = tf.add(tf.matmul(act2, weights['out']),biases['out'])
        return out  



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
        act1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        act1 = tf.nn.relu(act1)

        # Output layer with linear activation
        out = tf.add(tf.matmul(act1, weights['out']),biases['out'])
        return out  




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
        act1= tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        act1= tf.nn.relu(act1)
        #layer_1 = tf.nn.elu(layer_1)
        # Hidden layer with RELU activation
        act2 = tf.add(tf.matmul(act1, weights['h2']), biases['b2'])
        act2 = tf.nn.relu(act2)
        #layer_2 = tf.nn.elu(layer_2)
        # Output layer with linear activation
        out = tf.add(tf.matmul(act2, weights['out']),biases['out'])
        return out  