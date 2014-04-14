server = True
import scipy.io as io
import matplotlib
if server:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import alfileformat as al
import theano
import tools
import numpy
import theano.tensor as T
from theano.tensor.nnet import conv
from collections import defaultdict, OrderedDict
import glob
import os
import scipy
import scipy.ndimage.filters as filters 
import sys
import math
import random

MAX_AL_FILES = 1
if server:
    MAX_AL_FILES = 2000
curr_sc_no = 5
if server:
    curr_sc_no = int(sys.argv[1])
    
max_to_show = 1    
neigh_size = 2.0
weight = 1.0
plot_power = 2.0

class UnaryPots(object):    
    def __init__(self, filenames, filter, part, curr_sc):
        # both together should not be empty
        assert len(filter) != 0
        assert len(filenames) != 0
        
        self.part = part
        self.part_to_part_no = {'fac':0, 'sho':1, 'elb':2, 'wri':3}
        self.heatmaps = []
        self.scale = curr_sc
        
        filtered_filenames = self.__filter_filenames(filenames, filter)
        basefolder = '/'.join(filtered_filenames[0].split('/')[0:-1])
        for filename in filtered_filenames:
            striped_fname = filename.split('/')[-1].split('.')[0]
            assert filename.endswith('.npy')
            img = numpy.load(basefolder + '/' + striped_fname + '.npy')
            
            data_max = filters.maximum_filter(img, neigh_size)
            maxima = (img == data_max)
            img[maxima == False] = 0

            self.heatmaps.append(img)
    
    def __filter_filenames(self, filenames, filter):
        filtered_filenames = []
        basefolder = '/'.join(filenames[0].split('/')[0:-1])
        ext = filenames[0].split('.')[-1]
        for fname in filter:
            if any(fname in s for s in filenames):
                filtered_filenames.append(basefolder + '/' + fname + '.' + ext)
        return filtered_filenames

        

class StichHeapMaps(object):
        
    def __init__(self):
        margin = 125
        self.sho_given_elb = plt.imread('../data/priors/sho_given_elb.png')[margin:-margin, margin:-margin]
        self.elb_given_wri = plt.imread('../data/priors/elb_given_wri.png')[margin:-margin, margin:-margin]
        self.fac_given_sho = plt.imread('../data/priors/fac_given_sho.png')[margin:-margin, margin:-margin]
        
        # TODO: Play with shrink / dialate filters here.
        
#         self.elb_given_wri[20:110, 28:120] = 1.0
#         self.sho_given_elb[8:70, 35:90] = 1.0
#         self.fac_given_sho[30:65, 30:65] = 1.0

        
        self.face_global_prior_torso_space = plt.imread('../data/priors/face_global_prior_torso_space.png')
        
        self.wri_given_elb = numpy.rot90(self.elb_given_wri, 2)
        self.elb_given_sho = numpy.rot90(self.sho_given_elb, 2)
        self.sho_given_fac = numpy.rot90(self.fac_given_sho, 2)
        
        
        # self.parts = ['sho', 'elb', 'wri']
        self.HMap = defaultdict(list)
        
    def load_data(self, dirs, scale=1.0):
        # convolve the local maximas with the filters
        filter = []
        unaries = []
        parts, dirpaths = dirs
        
        filenames = glob.glob(dirpaths[0])
        # random.shuffle(filenames)
        self.master_filter = set([filename.split('/')[-1].split('.')[0] for filename in filenames])
        self.master_filter = set(list(self.master_filter)[0:MAX_AL_FILES])
        for idx in range(1, len(parts)):
            curr_filter = set([filename.split('/')[-1].split('.')[0] for filename in glob.glob(dirpaths[idx])]) 
            self.master_filter = self.master_filter & curr_filter if len(curr_filter) > 0 else self.master_filter
            
        for idx in range(0, len(parts)):
            filenames = glob.glob(dirpaths[idx])
            unary = UnaryPots(filenames, self.master_filter, parts[idx], scale)
            unaries.append(unary)
        
        # create and populate numpy tensor that can be given to theano
        for unary in unaries:
            self.HMap[unary.part] = numpy.ndarray(shape=(len(self.master_filter), 1, \
                                        unary.heatmaps[0].shape[0], unary.heatmaps[0].shape[1]), dtype=theano.config.floatX)
            for idx, hmap in enumerate(unary.heatmaps):
                self.HMap[unary.part][idx][0] = hmap
        
        if not server:
            for i, fname in enumerate(self.master_filter):
                for part in ['fac', 'sho', 'elb', 'wri']:
                    max_loc = numpy.unravel_index(self.HMap[part][i, 0, :, :].argmax(), self.HMap[part][i, 0, :, :].shape)
                    plt.imshow(self.HMap[part][i, 0, :, :])
                    plt.title(part)
                    plt.imsave('/home/user/MODEC/cropped-images/hmap_nosm/' + fname + '_' + part + '.png', self.HMap[part][i, 0, :, :])
                    plt.show()
                    im = plt.imread('/home/user/MODEC/cropped-images/' + fname + '.png')
                    plt.imshow(im)
                    plt.scatter((max_loc[1]) * 4.0 / scale, (max_loc[0]) * 4.0 / scale)
                    plt.title(part)
                    plt.show()

    
    def multiply_with_unary(self, input, y):
        # mutilply local evedence
        local_evidence = self.HMap[y]
        # TODO: make sure this is elementwise only
        assert input.min() >= 0
        return input * local_evidence
    
    def multiply_fac_with_glob_prior(self):
        glob_evidence = self.face_global_prior_torso_space
        curr_scale = self.HMap['fac'][0][0].shape
        
        glob_evidence = numpy.asarray(scipy.misc.imresize(glob_evidence, curr_scale), dtype=theano.config.floatX)
        glob_evidence /= glob_evidence.max()
        
        assert self.HMap['fac'].shape[2] == glob_evidence.shape[0] and self.HMap['fac'].shape[3] == glob_evidence.shape[1]
        assert glob_evidence.min() >= 0
        for i0 in range(self.HMap['fac'].shape[0]):  # for so many images
            self.HMap['fac'][i0][0] = self.HMap['fac'][i0][0] * glob_evidence
        
    def compute_y_given_x(self, y, x, input=None): 
        print 'activ_filter: ' + y + '_given_' + x
        activ_filter = getattr(self, y + '_given_' + x)
        activ_filter = activ_filter ** weight
        if input == None:
            input = self.HMap[x]
        
        X = T.matrix(dtype=theano.config.floatX)
        X = X.reshape(input.shape)
        filter_shape = (1, 1, activ_filter.shape[0], activ_filter.shape[1])
        filters = numpy.asarray(activ_filter.reshape(filter_shape), dtype=theano.config.floatX)
    
        convout = conv.conv2d(input=X,
                                 filters=filters,
                                 image_shape=input.shape,
                                 filter_shape=filter_shape,
                                 border_mode='full')
    
        # For each pixel, remove mean of 9x9 neighborhood
        mid = numpy.asarray(numpy.floor(numpy.asarray(activ_filter.shape) / 2.), dtype=int)
        # TODO: make sense of this +1
        convout = convout[:, :, mid[0]:-mid[0] + 1, mid[1]:-mid[1] + 1]
        f = theano.function([X], convout)
        output = f(input)
        return output       


        
scales = [0.44 + i0 * 0.1 for i0 in range(6)]
keys = ['wri', 'elb', 'sho', 'fac']
epoch = 400
scale = scales[curr_sc_no]
file = '*.npy'
# file = 'american-wedding-unrated6x9-00086711*.npy'
vals = ['/home/user/Projects/deep_nets/fullset_multiscale_heatmap_negmoff/' + part + '_dec27/FLIC-sapp-all_sc_' + str(scale) + '_ep_' + str(epoch) + '/' + file for part in keys]
print vals
shm = StichHeapMaps()
shm.load_data(dirs=(keys, vals), scale=scale)



def plot_maximas_on_image(distribution):
    # imgname = '/Users/ajain/Projects/MODEC/cropped-images/12-oclock-high-special-edition-00171221_00141.png'
    # im = plt.imread(imgname)
    plt.imshow(distribution)
    rows, cols = numpy.nonzero(distribution)
    xs = []
    ys = []
    ss = []
    for idx in range(0, rows.shape[0]):
        y = rows[idx] 
        x = cols[idx] 
        score = distribution[y, x]
        xs.append(x)
        ys.append(y)
        ss.append(score)
    # plt.scatter(xs, ys, c=ss, cmap=cm.coolwarm, s=5)
    plt.show()



print 'Global face prior'
g_img = scipy.ndimage.interpolation.zoom(shm.face_global_prior_torso_space, scale)
shm.multiply_fac_with_glob_prior()  # note this changes the unary itself in the class

print 'Sho / Face prior'
op_sho_given_fac = shm.compute_y_given_x(y='sho', x='fac')
 
print 'Sho Unary * Sho/Face Prior'
op_sho_given_fac_sho_un = op_sho_given_fac * shm.HMap['sho']

print 'Elb / Sho prior'
op_elb_given_sho = shm.compute_y_given_x(y='elb', x='sho', input=op_sho_given_fac_sho_un)

print 'Elb Unary * Elb/Sho Prior'
op_elb_given_sho_elb_un = op_elb_given_sho * shm.HMap['elb']

print 'Wri / Elb prior'
op_wri_given_elb = shm.compute_y_given_x(y='wri', x='elb', input=op_elb_given_sho_elb_un)
     
print 'Wri Unary * Wri/Elb Prior'
op_wri_given_elb_wri_un = op_wri_given_elb * shm.HMap['wri']

print 'Elb / Wri prior'
op_elb_given_wri = shm.compute_y_given_x(y='elb', x='wri')

print 'Elb Unary * Elb/Wri'
op_elb_given_wri_elb_un = op_elb_given_wri * shm.HMap['elb']

print 'Sho / Elb prior'
op_sho_given_elb = shm.compute_y_given_x(y='sho', x='elb', input=op_elb_given_wri_elb_un)

print 'Sho'
op_sho = op_sho_given_fac * shm.HMap['sho']

print 'Elb'
op_elb = op_elb_given_wri * op_elb_given_sho * shm.HMap['elb']

print 'Wri'
op_hmap_path_prefix = '../tmp/'
oppath = op_hmap_path_prefix + '/' + 'beforesp_sc{0}'.format(curr_sc_no)
numpy.save(oppath, shm.HMap['wri'][0, 0, :, :])
op_wri = op_wri_given_elb * shm.HMap['wri']
oppath = op_hmap_path_prefix + '/' + 'aftersp_sc{0}'.format(curr_sc_no)
numpy.save(oppath, op_wri[0, 0, :, :])

# plt.imshow(shm.HMap['wri'][0,0,:,:])
# plt.show()
# plt.imshow(op_wri[0,0,:,:])
# plt.show()
# print numpy.fabs(((op_wri - shm.HMap['wri']))).max()
# diff = numpy.fabs(((op_wri - shm.HMap['wri'])))
# plt.imshow(diff[0,0,:,:])
# plt.title('diff')
# plt.show()
# plt.imshow(op_wri_given_elb[0,0,:,:])
# plt.title('op_wri_given_elb')
# plt.show()


print 'Face'
op_fac = shm.HMap['fac'] 

distributions = {'wri':op_wri, 'elb':op_elb, 'sho':op_sho, 'fac':op_fac}
for part in ['wri', 'elb', 'sho', 'fac']:
	op_dir = '/'.join(vals[0].split('/')[0:-3]) + '/' + part + '/stitched_sc_' + str(scale) + '/'
	if not os.path.exists(op_dir):
    		print 'mkdir: ' + op_dir
    		os.makedirs(op_dir)

	for i, fname in enumerate(shm.master_filter):
		distribution = distributions[part]
		max_loc = numpy.unravel_index(distribution[i, 0, :, :].argmax(), distribution[i, 0, :, :].shape)
		jnt_pos_2d = dict()
		jnt_pos_2d[fname + '.png'] = [(max_loc[1] * 4.0 / scale, max_loc[0] * 4.0 / scale, distribution[i, 0, max_loc[0], max_loc[1]])]
		al.write_many(jnt_pos_2d, op_dir + fname + '.al')
        
        if not server:
            plt.imshow(distribution[i, 0, :, :])
            plt.title(part)
            plt.imsave('/home/user/MODEC/cropped-images/hmap_sm/' + fname + '_' + part + '.png', distribution[i, 0, :, :])
            plt.show()

