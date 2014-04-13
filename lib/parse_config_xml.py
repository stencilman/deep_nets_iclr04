import xml.etree.ElementTree as ET  
import tools
import shutil
import os, sys
import re

class ParseConfigXML(object): 
    def __init__(self, machinefile):        
        self.conf_file = machinefile
        self.__parse_conffile()
        
        # to copy parameters, so that we know which file is what.

        if not os.path.exists(self.op_dir):
            print 'mkdir: ' + self.op_dir
            os.makedirs(self.op_dir)
        try:
            shutil.copy2(self.conf_file, self.op_dir + re.split(r'[\\/]+', self.conf_file)[-1])
        except Exception, e: 
            print str(e)

        if not os.path.exists(self.shared_op_dir):
            print 'mkdir: ' + self.shared_op_dir
            os.makedirs(self.shared_op_dir)
        
    def __parse_conffile(self):
        self.conf = ET.parse(self.conf_file)  

        
        self.cost = 'L2'
        if self.conf.find('cost') is not None:
            self.cost = self.conf.find('cost').text
        
        pnt_nos = self.conf.find('pnt_no').text.split(',')
        self.pnt_nos = [int(p) for p in pnt_nos]

        self.op_dir = self.conf.find('logging/output_dir').text + '/'
        self.shared_op_dir = self.conf.find('logging/shared_output_dir').text + '/'
        self.weights_dir = self.conf.find('init/weights_dir').text + '/'
    
        self.load_weights = False
            
        self.learning_rate = float(self.conf.find('learning_rate').text)
        self.n_iters = int(self.conf.find('n_iters').text)
        self.n_epochs = int(self.conf.find('n_epochs').text)
 
        self.batch_size = int(self.conf.find('batch_size').text)
        self.thresh = int(self.conf.find('test_thresh').text)
        self.test_dir = self.conf.find('test_dir').text
        self.reg_weight = float(self.conf.find('reg_weight').text)       
        self.momentum = float(self.conf.find('momentum').text)   
        
        if self.momentum < sys.float_info.epsilon:
            self.use_momentum = False 
        else:
            self.use_momentum = True
        
        self.rmsprop_filter_weight = float(self.conf.find('rmsprop_filter_weight').text) 
        self.rmsprop_maxgain = float(self.conf.find('rmsprop_maxgain').text) 
        self.rmsprop_mingain = float(self.conf.find('rmsprop_mingain').text) 
        
        if self.rmsprop_filter_weight < sys.float_info.epsilon:
            self.use_rmsprop = False 
        else:
            self.use_rmsprop = True
            
        self.hard_mine_freq = int(self.conf.find('hard_mine_freq').text)   
        self.epoch_no = int(self.conf.find('epoch_no').text)   

        self.log_filename = self.conf.find('log_filename').text
        
        self.mix_ratio = dict()
        for kind in ['train', 'test']:
            self.mix_ratio[kind] = self.conf.find('mix_ratio/' + kind).text.split(':')
            self.mix_ratio[kind] = float(self.mix_ratio[kind][0]) / float(self.mix_ratio[kind][1])
        
        self.perturb = False
        self.perturb = self.conf.find('perturb').text == 'True'
        
        
