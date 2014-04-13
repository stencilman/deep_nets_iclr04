import random

import theano
import theano.tensor as T

import numpy
import scipy
import scipy.misc as misc
import scipy.io as io
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

import re
import time
import sys, os 

import xml.etree.ElementTree as ET  
import cPickle as pickle
import glob
from collections import defaultdict

# My own libs
import tools
import alfileformat as al
from layer_blocks import ConvPoolLayer, OutputLayer, InputLayerSW, \
    SoftMaxLayer, SVMLayer, DropoutLayer

class Machine(object):    
    layer_map = {
      'InputLayerSW': InputLayerSW,
      'ConvPoolLayer' : ConvPoolLayer,
      'OutputLayer' : OutputLayer,
      'SoftMaxLayer': SoftMaxLayer,
      'SVMLayer' : SVMLayer,
      'DropoutLayer' : DropoutLayer,
      }
    
    def __init__(self, params):
        # argv[1] is the conf xml file
        # set extra / overwrite params
        if len(sys.argv) > 2:
            print sys.argv[2];
            type_map ={'int':int, 'bool':bool, 'str':str, 'float':float}
            test_param_str = sys.argv[2].split(',')
            print test_param_str
            for t_par in test_param_str:
                print t_par
                key, val = t_par.split('=')
                print key, val
                val, type = re.split('\(*\)*', val)[0:2]
                print 'val={0}, type={1}'.format(val, type)
                val = type_map[type](val)
                print 'Setting params.{0} to {1}'.format(key, val)
                setattr(params, key, val) 