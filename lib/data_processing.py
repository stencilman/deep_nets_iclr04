import os, sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../lib'))
if not path in sys.path:
    sys.path.insert(1, path)
import matplotlib.pyplot as plt
import tools
import json
from collections import defaultdict 
import time
import scipy.io as io
import scipy.misc as misc
import numpy
import tools
import cPickle as pickle
import shlex
import subprocess as sp
import time
import json
import xml.etree.ElementTree as ET
import random
import scipy.io
import alfileformat as al
import socket

random.seed(857)

import warnings
warnings.filterwarnings("ignore")

        
                    
class DatasetProcessFLIC(object):
    def __init__(self, type='FLIC'):
        if type == 'FLIC':
            ip_dir = 'cropped-images'
        elif type == 'SHOULDER':
            ip_dir = 'full-images'
        
        self.ptno_part = {0:'face', 1:'lsho', 2:'lelb', 3:'lwri', 4:'rsho', 5:'relb', 6:'rwri'}
        self.part_pos = dict()
        for pt_no, part in self.ptno_part.items():
            matname = self.ptno_part[pt_no] + '_pos.mat'
            matkey =  self.ptno_part[pt_no] + 'Pos'
            self.part_pos[part] = io.loadmat('unprocessed_data/'+ip_dir+'/' + matname)[matkey]
        
        self.names = io.loadmat('unprocessed_data/'+ip_dir+'/names.mat')['nameList'][0]
        self.is_train = io.loadmat('unprocessed_data/'+ip_dir+'/istrain.mat')['train_set'][0]
        self.scale_and_crop_coords = io.loadmat('unprocessed_data/'+ip_dir+'/scale_and_crop_coords.mat')['scale_and_crop_coords'][0]
                        
        self.X = defaultdict(list)
        self.Y = defaultdict(list)
        self.index = defaultdict(list)
        
        
        # which file is train, test, valid
        # no validation
        train_valid_sep = 10000
        X_names = defaultdict(list)      
        for idx in range(0, len(self.names)):
            if self.is_train[idx] == 1 and len(X_names['train']) < train_valid_sep:
                X_names['train'].append(self.names[idx])
                self.index['train'].append(idx)
            elif self.is_train[idx] == 1 and len(X_names['train']) >= train_valid_sep:   
                X_names['valid'].append(self.names[idx])
                self.index['valid'].append(idx)
            else:
                self.index['test'].append(idx)
                X_names['test'].append(self.names[idx])
        
        test_indices_subset  = [170, 171, 172, 173, 174, 175, 176, 376, 377, 378, 379, 380, 381, 384, 386, 389, 390, 391, 392, 393, 394, 398, 400, 401, 402, 404, 405, 407, 408, 417, 699, 700, 701, 702, 703, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 733, 734, 735, 752, 754, 755, 756, 757, 896, 897, 898, 899, 900, 903, 904, 905, 906, 907, 918, 919, 920, 961, 963, 964, 965, 966, 967, 981, 982, 983, 1526, 1527, 1528, 1529, 1533, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1561, 1576, 1577, 1609, 1610, 1611, 1612, 1613, 1614, 1626, 1627, 1777, 1778, 1779, 1780, 1781, 1783, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1815, 1856, 1857, 1858, 1859, 1860, 1885, 2324, 2325, 2327, 2328, 2329, 2330, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2950, 2952, 2953, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2969, 2970, 2971, 2972, 2973, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3277, 3278, 3279, 3280, 3285, 3286, 3287, 3288, 3300, 3305, 3341, 3344, 3345, 3389, 3390, 3391, 3392, 3393, 3395, 3397, 3398, 3592, 3593, 3594, 3595, 3596, 3597, 3625, 3768, 3769, 3770, 3771, 3772, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3845, 3846, 3847, 3848, 3849, 3850, 3884, 3961, 3962, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4376, 4382, 4390, 4395, 4396, 4397, 4406, 4407, 4584, 4585, 4586, 4787, 4790, 4792, 4793, 4796, 4812, 4813, 4814, 4815, 4816, 4817, 4818, 4967, 4968, 4969, 4981, 4982, 4995, 4996, 4997, 4998, 4999, 5000, 5001, 5002, 5003]
        test_indices_subset[:] = [x - 1 for x in test_indices_subset]
        X_names['test'] = [self.names[i] for i in test_indices_subset]
        self.index['test'] = test_indices_subset
        print test_indices_subset
        
        #load x and y in memory
        for kind in ['train', 'valid', 'test']:
            self.X[kind] = [None] * len(X_names[kind])
            self.Y[kind] = [None] * len(X_names[kind])
         
        for kind in ['train', 'valid', 'test']:
            for idx, name in enumerate(X_names[kind]): 
                im = plt.imread('unprocessed_data/'+str(name[0]))
                if socket.gethostname() != 'vajra' and sys.platform != 'darwin':
                    im = misc.imrotate(im, 180.0)
                    im = numpy.fliplr(im)               
                self.X[kind][idx] = im
                
                self.Y[kind][idx] = []
                for pt_no, part in self.ptno_part.items():
                    self.Y[kind][idx].append((self.part_pos[part][0][self.index[kind][idx]], self.part_pos[part][1][self.index[kind][idx]]))
                if idx % 100 == 0:
                    print '{0:d} / {1:d}'.format(idx, len(self.names))
        
        for kind in ['train', 'valid', 'test']:
            print 'no of {0:s}: {1:d}'.format(kind, len(self.X[kind]))
        
        if type == 'SHOULDER':
            self.scale_and_crop_images()
        
        #flip train and valid
        for kind in ['train', 'valid']:
            for idx in range(0, len(self.X[kind])):
                flipped = numpy.fliplr(self.X[kind][idx])
                flip_name = '.'.join(X_names[kind][idx][0].split('.')[0:-1])+'-flipped.jpg'
                self.X[kind].append(flipped)
                X_names[kind].append([flip_name])
                flip_y = [(flipped.shape[1] - self.Y[kind][idx][pt_no][0], self.Y[kind][idx][pt_no][1]) for pt_no in self.ptno_part.keys()]
                remapped = [flip_y[j0] for j0 in [0, 4, 5, 6, 1, 2, 3]]
                flip_y = remapped
                self.Y[kind].append(flip_y)                
                
        for kind in ['train', 'valid', 'test']:
            print 'no of {0:s}: {1:d}'.format(kind, len(self.X[kind]))
          
        #------- Write it all down to disk -------#
        target_imgshape = (240, 320, 3)
        scalefactor = float(target_imgshape[0])/self.X['train'][0].shape[0] 
        print 'Image shape: '
        print self.X['train'][0].shape
        print 'Scalefactor is: ' + str(scalefactor)
        for kind in ['train', 'valid', 'test']:
            jnt_pos_2d = dict()
            print 'writing images for '+ kind
            if not os.path.exists('processed_data/'+ kind):
                os.makedirs('processed_data/' + kind)
            p_imname_tmpl = 'processed_data/' + kind + '/{0:s}.png'    
            for idx in range(0, len(self.X[kind])):  
                if idx % 100 == 0:
                    print '{0:d} / {1:d}'.format(idx, len(self.X[kind]))  
                scaled_im = misc.imresize(self.X[kind][idx], target_imgshape)  
                imname = X_names[kind][idx]
                imname = imname[0].split('/')[-1].split('.')[0]
                imname = p_imname_tmpl.format(imname)
                misc.imsave(imname, scaled_im)
                
                for pt_no, part in self.ptno_part.items():
                    x = self.Y[kind][idx][pt_no][0] * scalefactor 
                    y = self.Y[kind][idx][pt_no][1] * scalefactor 
                    if imname not in jnt_pos_2d.keys():
                        jnt_pos_2d[imname] = [(x, y)]
                    else:
                        jnt_pos_2d[imname].append((x, y)) 
                """
                plt.imshow(scaled_im)
                xs = [jnt[0] for jnt in jnt_pos_2d[imname]]
                ys = [jnt[1] for jnt in jnt_pos_2d[imname]]
                plt.scatter(xs, ys)
                plt.show()
                """
                 
                    
            tools.pickle_dump(jnt_pos_2d, 'processed_data/' + kind + '/jnt_pos_2d.pkl')
            al.write(jnt_pos_2d, 'processed_data/' + kind + '/jnt_pos_2d.al')        
        
    def scale_and_crop_images(self):
        #{0:'face', 1:'lsho', 2:'lelb', 3:'lwri', 4:'rsho', 5:'relb', 6:'rwri'}
        total = 0     
        off_x = [-1] * 5003
        off_y = [-1] * 5003
        f_scale = [-1] * 5003
        
        pad_size = 200
        for kind in ['train', 'valid', 'test']:
            for i0 in range(len(self.X[kind])):
                img = self.X[kind][i0]
                scale = self.scale_and_crop_coords[self.index[kind][i0]][1][0][0]*1.2
        
                img = misc.imresize(img, scale)
                img = tools.pad_allsides(img, pad_size, border='black')
                
                ref_pt_x = (self.Y[kind][i0][1][0] + self.Y[kind][i0][4][0])*scale/2.0 + pad_size
                ref_pt_y = (self.Y[kind][i0][1][1] + self.Y[kind][i0][4][1])*scale/2.0 + pad_size
                
                img = img[ref_pt_y-80: ref_pt_y + 160, ref_pt_x - 160: ref_pt_x + 160, :]
                self.X[kind][i0] = img
                offset_x = 160 - ref_pt_x
                offset_y = 80 - ref_pt_y
                if kind == 'test':
                    img_idx = self.index[kind][i0]
                    off_x[img_idx] = ref_pt_x - 160
                    off_y[img_idx] = ref_pt_y - 80
                    f_scale[img_idx] = scale
                
                
                xs = [self.Y[kind][i0][pnt][0] * scale + pad_size + offset_x for pnt in self.ptno_part.keys()]
                ys = [self.Y[kind][i0][pnt][1] * scale + pad_size + offset_y for pnt in self.ptno_part.keys()]
                for pnt in self.ptno_part.keys():
                    self.Y[kind][i0][pnt] = (xs[pnt], ys[pnt])
                    
            #io.savemat('off_x.mat', dict(off_x=numpy.asarray(off_x)) )
            #io.savemat('off_y.mat', dict(off_y=numpy.asarray(off_y)))
            #io.savemat('f_scale.mat', dict(f_scale=numpy.asarray(f_scale)))
            

        
                                
            
