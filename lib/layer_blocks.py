import sys, os

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import re
import time
import cPickle as pickle
from types import *

import numpy
import math 
import matplotlib.pyplot as plt

from data_handling import DataSlidingWindow 
import tools

import warnings
warnings.filterwarnings("ignore")

rng = numpy.random.RandomState(23455)  #constant random seed: such that experiments are repeatable
        
class Layer(object):
    instance_count = 0
    def __init__(self, layerxml, params):
        Layer.instance_count += 1
        if layerxml.find('id') != None:
            self.id = layerxml.find('id').text
        else:
            self.id = str(Layer.instance_count)
            
        self.layer_no = Layer.instance_count
        self.log = False
        self.representative_image = None
        
        self.load_weights_while_train = False
        if layerxml.find('load_weights') != None and  layerxml.find('load_weights').text == 'True':
            self.load_weights_while_train = True
            
        self.load_weights = False 
        if params.load_weights or (layerxml.find('load_weights') != None and  layerxml.find('load_weights').text == 'True'):
            self.load_weights = True 
        
        self.log = False
        if layerxml.find('log') is not None:
            self.log = layerxml.find('log').text  == 'True'
        
        self.weight_update = True
        if layerxml.find('weight_update') is not None:
            self.weight_update = layerxml.find('weight_update').text  == 'True'
        print 'Weight update for layer {0:s} is {1:s}'.format(self.id, str(self.weight_update))
        
        # The default output and input size is 'empty'.  
        # - All sub-classes are expected to set the output size to something 
        #   that is not empty in their __init__ method.
        # - The convention is that the 0th dimension is the batch size.
        self.out_size = (0)  # Default output size is 'empty'
        self.in_size = (0)  # Default output size is 'empty'

class Weight(object):      
    def __init__(self, w_shape, load_weights, weights_dir, bound, name, epoch): 
        super(Weight, self).__init__()  
        self.name = name
	if not load_weights:
            self.np_values = numpy.asarray(bound * rng.standard_normal(w_shape), dtype=theano.config.floatX)   
        else:
            self.load_weight(weights_dir, epoch)
        
        if type(w_shape) == IntType:
            self.np_values = self.np_values + math.fabs(self.np_values.min())
        self.val = theano.shared(value=self.np_values, name=name) 

    def save_weight(self, dir, epoch = 0):
        print '- Saved weight: ' + self.name + '_ep'+str(epoch)
        numpy.save(dir + self.name + '_ep'+str(epoch), self.val.get_value())
    
    def load_weight(self, dir, epoch = 0):
        print '- Loaded weight: ' + self.name + '_ep'+str(epoch)
        self.np_values = numpy.load(dir + self.name + '_ep'+str(epoch) + '.npy')
        

class SoftMaxLayer(Layer):
    type = 'SoftMaxLayer'

    def  __init__(self, layerxml, params, prev_layer = None):
        super(SoftMaxLayer, self).__init__(layerxml, params)
        
        if prev_layer.type != "OutputLayer" and prev_layer.type != "DropoutLayer":
            raise NotImplementedError()
        if len(prev_layer.out_size) != 2:
            raise NotImplementedError()
        
        self.in_size = (params.batch_size, prev_layer.out_size[-1])
        self.out_size = (params.batch_size, 2)
        
        n_in = self.in_size[-1]
        n_out = self.out_size[-1]
                                 
        print 'No of input units: ' + str(n_in)
        print 'No of output units: ' + str(n_out)
        
        W_bound = 0.00000000  #numpy.sqrt(6. / (n_in + n_out))
        self.W = Weight((n_in, n_out), self.load_weights, params.weights_dir, \
                        W_bound, 'W_' + str(self.id), params.epoch_no)
        self.b = Weight(n_out, self.load_weights, params.weights_dir, \
                        W_bound, 'b_' + str(self.id), params.epoch_no)
        
    def compute(self, input, params):
        input = input.flatten(2)
        self.input = input       
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W.val) + self.b.val)
        self.output = T.argmax(self.p_y_given_x, axis=1)      
        # confidence of the label
        self.confidence = self.p_y_given_x[T.arange(self.p_y_given_x.shape[0]), self.output]        
        self.params = [self.W.val, self.b.val]
        
    def write_log(self, params, get_output_layer, epoch, iter):   
        # write layer1 output                  
        op_out = numpy.asarray([get_output_layer(0)]).T
        l0name = '{0:s}/SofMax-test-epoch{1:04d}-iter{2:04d}-'.format(params.op_dir, epoch, iter) + '{0:d}.png'
        tools.write_image_multiple(op_out, (1,1), params.pnt_nos, l0name)

class OutputLayer(Layer):
    type = 'OutputLayer'
    
    def  __init__(self, layerxml, params, prev_layer = None):
        super(OutputLayer, self).__init__(layerxml, params)
        
        self.in_size = prev_layer.out_size
        self.out_size = (params.batch_size, int(layerxml.find('no_states').text))
        
        n_in = numpy.prod(prev_layer.out_size[1:])
        n_out = self.out_size[-1]
                    
        print 'No of input units: ' + str(n_in)
        print 'No of output units: ' + str(n_out)
        
        W_bound = 1.0/numpy.sqrt(n_in)
        if layerxml.find('activation') is not None:
            if layerxml.find('activation').text == 'tanh':
                self.activation = T.tanh 
            elif layerxml.find('activation').text == 'relu':
                self.activation = self.relu
        else:
            self.activation = T.nnet.sigmoid
            W_bound *= 4    
            
        self.W = Weight((n_in, n_out), self.load_weights, params.weights_dir, W_bound, 'W_' + str(self.id), params.epoch_no)
        self.b = Weight(n_out, self.load_weights, params.weights_dir, W_bound, 'b_' + str(self.id), params.epoch_no)
    
    def relu(self, x):
        return T.maximum(x, 0)
    
    def compute(self, input, params):
        input = input.flatten(2)
        self.input = input 
        lin_output = T.dot(self.input, self.W.val) + self.b.val
        self.output = self.activation(lin_output)
        #self.output = T.nnet.sigmoid(lin_output)
        # parameters of the model
        self.params = [self.W.val, self.b.val]
        
    def write_log(self, params, get_output_layer, epoch, iter):   
        # write layer1 output                  
        op_out = get_output_layer(0)
        l0name = '{0:s}/Output-test-epoch{1:04d}-iter{2:04d}-'.format(params.op_dir, epoch, iter) + '{0:d}.png'
        #tools.write_image_multiple(op_out, params.imshape[:2], params.pnt_nos, l0name)
        #self.representative_image = l0name
        tools.write_image_multiple(op_out, (1,1), params.pnt_nos, l0name)
        
class DropoutLayer(Layer):
    type = 'DropoutLayer'
    # Each layer instance needs its own seed, so draw from this srng to get the 
    # seeds for each layer.
    __dropout_seed_srng = numpy.random.RandomState(0)
    # In order to turn dropout off at test time we need to keep track of the
    # probability of all our dropout layers that have been instantiated.
    dropout_layers = []
    
    def  __init__(self, layerxml, params, prev_layer = None):
        super(DropoutLayer, self).__init__(layerxml, params)
        
        DropoutLayer.dropout_layers.append(self)  # TODO: Is this threadsafe?
        
        # TODO: Make dropout probability programmable (ie, use a shared variable)
        self.prob = float(layerxml.find('prob').text)
                    
        print 'Dropout Probability: ' + str(self.prob)
        
        self.in_size = prev_layer.out_size
        self.out_size = self.in_size
    
    def compute(self, input, params):   
        # We need to be able to turn on and off the dropout (on for training, 
        # off for testing).  Therefore use a shared variable to control
        # the current dropout state.  Start in "ON" state by default.
        self.dropout_on = T.shared(numpy.cast[theano.config.floatX](1.0), \
                                   borrow=True)
        
        # Create a random stream to generate a random mask of 0 and 1
        # activations.
        seed = DropoutLayer.__dropout_seed_srng.randint(0, sys.maxint)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        # p=1-p because 1's indicate keep and p is prob of dropping
        self.mask = srng.binomial(n=1, p=1.0 - self.prob, size=input.shape)
        
        # When dropout is off, activations must be multiplied by the average
        # on probability (ie 1 - p)
        off_gain = (1.0 - self.prob)
        # The cast in the following expression is important because:
        # int * float32 = float64 which pulls things off the gpu
        self.output = input * self.dropout_on * T.cast(self.mask, theano.config.floatX) + \
            off_gain * input * (1.0 - self.dropout_on)
    
    # Static method to turn off dropout for all DropoutLayer instances
    # When training set training to True, otherwise False
    @staticmethod
    def SetDropoutOn(training):
        if training:
            dropout_on = 1.0
        else:
            dropout_on = 0.0
        for i in range(0, len(DropoutLayer.dropout_layers)):
            DropoutLayer.dropout_layers[i].dropout_on.set_value(dropout_on)
    
class InputLayerSW(Layer):
    type = 'InputLayerSW'
    
    def __init__(self, layerxml, params, prev_layer = None, windowshape = None):
        super(InputLayerSW, self).__init__(layerxml, params)
        
        # InputLayerSW must be the first layer
        if prev_layer != None:
            raise NotImplementedError()
        
        self.data_info = []   
        for d in layerxml.findall('data'):
            data_dir = d.find('dir').text + '/'
            self.data_info.append((data_dir))
       
        self.prepro_type = layerxml.find('preprocessing').text
        self.windowshape = windowshape
        if self.windowshape == None:
            #TODO: Parse from layer_xml, not hardcode
            self.windowshape = (int(layerxml.find('windowshape/h').text), 
                                int(layerxml.find('windowshape/w').text), 
                                int(layerxml.find('windowshape/d').text))
                
        if not params.load_weights:
            self.data = DataSlidingWindow(points=params.pnt_nos)
            self.data.load_picked_data(params.shared_op_dir)
            for dir in self.data_info:
                print 'Adding  data from: ' + dir            
                self.data.add_to_dataset(dir)
            self.data.load_data(self.windowshape, shuffle=False)
            self.data.save_picked_data(params.shared_op_dir) 
        params.imshape = self.windowshape
        
        self.out_size = (params.batch_size, params.imshape[2], params.imshape[0], params.imshape[1])
        # TODO: Is this in_size correct?
        self.in_size = (params.batch_size, params.imshape[2], params.imshape[0], params.imshape[1])
        
        print("InputLayerSW out_size: " + str(self.out_size))
    
    def compute(self, input, params):
        self.output = input.reshape((params.batch_size, params.imshape[2], params.imshape[0], params.imshape[1]))  

    def write_log(self, params):
        # TODO: Fix this (I assume the data format was changed at some point)
        #tools.write_image(self.data.train_set_x.get_value(), (self.windowshape[0],self.windowshape[1], self.windowshape[2]), params.op_dir + 'train_x_before_preprocess.png')  
        #tools.write_image_multiple(self.data.train_set_y.get_value(), (1,1), params.pnt_nos, params.op_dir + 'train_y_slice-{0:d}.png')
        #tools.write_image(self.data.valid_set_x.get_value(), (self.windowshape[0],self.windowshape[1], self.windowshape[2]), params.op_dir + 'valid_x_before_preprocess.png')  
        #tools.write_image_multiple(self.data.valid_set_y.get_value(), (1,1), params.pnt_nos, params.op_dir + 'valid_y_slice-{0:d}.png')
        if len(self.data.X_names['test']) > 0:
            print 'Data format changed'
            #tools.write_image(self.data.X_SW_p['test'], (self.windowshape[0],self.windowshape[1], self.windowshape[2]), params.op_dir + 'X_SW_p.png')
            #tools.write_image(self.data.X_SW_p['test'], (self.windowshape[0],self.windowshape[1], self.windowshape[2]), params.op_dir + 'X_SW_n.png')
            #tools.write_image_multiple(self.data.test_set_y.get_value(), (1,1), params.pnt_nos, params.op_dir + 'test_set_y_slice-{0:d}.png')
    
class ConvPoolLayer(Layer):
    type = 'ConvPoolLayer'
    def __init__(self, layerxml, params, prev_layer = None):
        super(ConvPoolLayer, self).__init__(layerxml, params)
                
        self.nkerns = int(layerxml.find('nos_filters').text)
        self.filter_size = int(layerxml.find('filter_size').text)
        self.pool_size = int(layerxml.find('pool_size').text)
        self.log_output = bool(layerxml.find('log_output').text)
        
        # The input type to the ConvPoolLayer is restricted (we expect the input
        # to be an image, so only layers that have image inputs are allowed).
        if prev_layer.type != "ConvPoolLayer" and \
            prev_layer.type != "InputLayerSW" and \
            prev_layer.type != "DropoutLayer":
            raise NotImplementedError()
        
        # DropoutLayer is a special case.  It's output can be either 4D or 2D,
        # depending on what came before it.  Make sure this is correct.
        if len(prev_layer.out_size) != 4:
            raise NotImplementedError()
        
        poolsize=(self.pool_size, self.pool_size)
        
        self.in_size = prev_layer.out_size
        self.out_size = (params.batch_size, self.nkerns, self.in_size[2]/poolsize[0], self.in_size[3]/poolsize[1])
        
        print("ConvPoolLayer in_size: " + str(self.in_size))
        print("ConvPoolLayer out_size: " + str(self.out_size))
        
        # Filter shape is (#n_output_feats, #n_input_feats, filter_size, filter_size)
        self.filter_shape = (self.nkerns, self.in_size[1], self.filter_size, self.filter_size)
    
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(self.filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (self.filter_shape[0] * numpy.prod(self.filter_shape[2:]) / numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(1. / (fan_in)) 
        
        self.W = Weight(self.filter_shape, self.load_weights, params.weights_dir, W_bound, 'W_' + str(self.id), params.epoch_no)
        self.b = Weight(self.filter_shape[0], self.load_weights, params.weights_dir, W_bound, 'b_' + str(self.id), params.epoch_no)
        
    def compute(self, input, params):
        poolsize=(self.pool_size, self.pool_size)
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W.val, image_shape=self.in_size, border_mode='full')
        
        mid = numpy.asarray(numpy.floor(numpy.asarray(self.filter_shape[2:]) / 2.), dtype=int)
        conv_out = conv_out[:, :, mid[0]:-mid[0], mid[1]:-mid[1]]
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)


        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = pooled_out + self.b.val.dimshuffle('x', 0, 'x', 'x')
        self.output = T.maximum(self.output, 0) #T.nnet.sigmoid(self.output) #T.nnet.sigmoid(self.output)  #T.tanh(self.output) #
        
        # store parameters of this layer
        self.params = []
        if self.weight_update:
            self.params = [self.W.val, self.b.val]
        
    def write_log(self, params, get_output_layer, epoch, iter):  
        
        # write filters
        figname = '{0:s}/Filters-id{1:d}-epoch-{2:04d}-iter{3:04d}.png'.format(params.op_dir, self.layer_no-1, epoch, iter)
        filter_img = numpy.reshape(self.W.val.get_value()[:, 0, :, :], (self.nkerns, self.filter_size * self.filter_size))
        tools.write_image(filter_img, (self.filter_size, self.filter_size, 1), figname)
        self.representative_image = figname
        
        if self.log_output:
            # write conv-pool layer output
            idx_in_minibatch = 0;
            convname = '{0:s}/ConvOut-test-id{1:d}-epoch-{2:04d}iter{3:04d}.png'.format(params.op_dir, self.layer_no-1, epoch, iter)
            conv_out = get_output_layer(idx_in_minibatch)
            convpool_out_img = numpy.reshape(conv_out[0, :, :, :], (self.nkerns, conv_out.shape[2] * conv_out.shape[3]))
            after_pooling_imgshape = (conv_out.shape[2], conv_out.shape[3], 1)
            tools.write_image(convpool_out_img, after_pooling_imgshape, convname)
        
class SVMLayer(Layer):
    type = 'SVMLayer'
    """SVM-like layer
    """
    def __init__(self, layerxml, params, prev_layer = None):
        super(SVMLayer, self).__init__(layerxml, params)
        
        if prev_layer.type != "OutputLayer" and prev_layer.type != "DropoutLayer":
            raise NotImplementedError()
        if len(prev_layer.out_size) != 2:
            raise NotImplementedError()
        
        self.in_size = (params.batch_size, prev_layer.out_size[-1])
        self.out_size = (params.batch_size, 1)
        
        n_in = self.in_size[-1]
        n_out = self.out_size[-1]
                  
        print 'No of input units: ' + str(n_in)
        print 'No of output units: ' + str(n_out)

        W_bound = 0 #numpy.sqrt(6. / (self.n_in + self.n_out))
        self.W = Weight((n_in, n_out), self.load_weights, params.weights_dir, \
                        W_bound, 'W_' + str(self.id), params.epoch_no)
        self.b = Weight(n_out, self.load_weights, params.weights_dir, \
                        W_bound, 'b_' + str(self.id), params.epoch_no)
        
    def compute(self, input, params):
        #TODO: hack ???
        self.batch_size = params.batch_size
        input = input.flatten(2)
        self.input = input       
        self.p_y_given_x = T.dot(input, self.W.val) + self.b.val
        
        self.output = T.argmax(self.p_y_given_x, axis=1)
        
        self.confidence = self.p_y_given_x[T.arange(self.output.shape[0]), self.output]
        
        self.params = [self.W.val, self.b.val]
        
    def hinge(self, u):
            return T.maximum(0, 1 - u)

    def svm_cost(self, y1):
        """ return the one-vs-all svm cost
        given ground-truth y in one-hot {-1, 1} form """
        y1_printed = theano.printing.Print('this is important')(T.max(y1))
        margin = T.reshape(y1, (self.batch_size, 1)) * self.p_y_given_x
        cost = self.hinge(margin).mean(axis=0).sum()
        return cost


    def errors(self, y):
        """ compute zero-one loss
        note, y is in integer form, not one-hot
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    
    def write_log(self, params, get_output_layer, epoch, iter):   
        # write layer1 output                  
        op_out = numpy.asarray([get_output_layer(0)]).T
        l0name = '{0:s}/SVM-test-epoch{1:04d}-iter{2:04d}-'.format(params.op_dir, epoch, iter) + '{0:d}.png'
        tools.write_image_multiple(op_out, (1,1), params.pnt_nos, l0name)




        

