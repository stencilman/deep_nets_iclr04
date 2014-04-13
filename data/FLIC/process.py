import os, sys
import matplotlib
matplotlib.use('Agg')

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lib'))
if not path in sys.path:
    sys.path.insert(1, path)

import tools
import data_processing

if not os.path.exists('unprocessed_data'):
    os.makedirs('unprocessed_data')

tools.download_file_nyu('/home/ajain/data/Data/FLIC_data/train/full-images.zip', 'unprocessed_data/')

if not os.path.exists('unprocessed_data/full-images/the-departed-00207091_05003.jpg'):
    tools.run_command('unzip unprocessed_data/full-images.zip -d unprocessed_data')

    
data  = data_processing.DatasetProcessFLIC('SHOULDER')
