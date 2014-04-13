# USAGE:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python train_detector.py conf.xml

import matplotlib
import os, sys
if not sys.platform.startswith('win'):
    matplotlib.use('Agg')

import numpy
import numpy.random
import warnings
warnings.filterwarnings("ignore")

numpy.random.seed(42)
import random
random.seed(857)
    
path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib'))
if not path in sys.path:
    sys.path.append(path)
    
from parse_config_xml import ParseConfigXML
from train_machine import TrainMachine
from logger import Logger

def main():

    if len(sys.argv) < 2:
        sys.exit('Param 1 should be <machine.xml>')

    if not os.path.exists('./' + sys.argv[1]):
        sys.exit('ERROR: Machine %s was not found!' % sys.argv[1])
    
    # parse conf file and fill up params
    params = ParseConfigXML(sys.argv[1])
    params.class_name = 'TrainMachine'
    
    # Open the Logger object (will redirect stdout and stderr transparently)
    logger = Logger()
    logger.open(params.log_filename)
    
    # create training machine    
    machine = TrainMachine(params)
    # run machine    
    machine.compute(params)
    
if __name__ == '__main__':    
    main()
