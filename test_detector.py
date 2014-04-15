import matplotlib
matplotlib.use('Agg')

import numpy
import numpy.random
import warnings
warnings.filterwarnings("ignore")

numpy.random.seed(42)
import random
random.seed(857)

import os, sys
    
path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib'))
if not path in sys.path:
    sys.path.append(path)
    
from  parse_config_xml import ParseConfigXML
from test_machine import TestMachine


def main():

    if len(sys.argv) < 2:
        sys.exit('Param 1 should be <machine.xml>')

    if not os.path.exists('./' + sys.argv[1]):
        sys.exit('ERROR: Machine %s was not found!' % sys.argv[1])
    
    print 'Start ...'
    print sys.argv[1]
    # parse conf file and fill up params
    params = ParseConfigXML(sys.argv[1])
    params.load_weights = True
    params.class_name = 'TestMachineSW'
    
    # create testing machine    
    machine = TestMachine(params)
    # run machine    
    machine.compute(params)
    
if __name__ == '__main__':    
    main()
