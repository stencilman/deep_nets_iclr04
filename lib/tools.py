import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from numpy.lib import stride_tricks
from scipy import misc
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from scipy.special import * 
from pylab import *
import glob
import cPickle as pickle
import os
import re
import shlex
import subprocess as sp
import time
import theano.sandbox.cuda as cuda
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy

import warnings
warnings.filterwarnings("ignore")

compute_test_value = False  # Force computation of symbolic values
run_on_subset = False  # Use small test / training set for quick debugging
no_batches_on_gpu = 100
POSLABEL = 1
NEGLABEL = 0

def write_image(data, imshape, imgname, flat_type='rrggbb'):
    noimg_inrow = 15;
    noimg_incol = np.ceil(data.shape[0] / float(noimg_inrow)).astype('int32')
    space_bw_img = int(max(min(imshape[0], imshape[1]) * .1, 2))  # pixels
    # print 'Figure spacing = {0:d}'.format(space_bw_img)
    if len(imshape) == 2 or imshape[2] == 1:
        F = np.zeros((imshape[0] * noimg_incol + space_bw_img * (noimg_incol - 1), \
                      imshape[1] * noimg_inrow + space_bw_img * (noimg_inrow - 1))) + data.max() / 2
        imshape = imshape[0:2]
    else:
        F = np.zeros((imshape[0] * noimg_incol + space_bw_img * (noimg_incol - 1), \
                      imshape[1] * noimg_inrow + space_bw_img * (noimg_inrow - 1), imshape[2])) + data.max() / 2

    im_idx = 0
    for r in range(noimg_incol):
        for c in range(noimg_inrow):   
            if im_idx < data.shape[0]:
                if len(imshape) == 2 or imshape[2] == 1:
                    im = data[im_idx].reshape(imshape)
                else:
                    if flat_type == 'rrggbb':
                        r_ch = data[im_idx][0:np.prod(imshape[0:2])].reshape(imshape[0:2])
                        g_ch = data[im_idx][np.prod(imshape[0:2]):2 * np.prod(imshape[0:2])].reshape(imshape[0:2])
                        b_ch = data[im_idx][2 * np.prod(imshape[0:2]):].reshape(imshape[0:2])
                    elif flat_type == 'rgbrgb':
                        r_ch = data[im_idx][0::3].reshape(imshape[0:2])
                        g_ch = data[im_idx][1::3].reshape(imshape[0:2])
                        b_ch = data[im_idx][2::3].reshape(imshape[0:2])
                    im = np.dstack((r_ch, g_ch, b_ch))
                    
                F[r * (imshape[0] + space_bw_img):(r + 1) * (imshape[0]) + r * space_bw_img, \
                  c * (imshape[1] + space_bw_img):(c + 1) * (imshape[1]) + c * space_bw_img ] = im
                im_idx += 1
                
    misc.imsave(imgname, F)
    # global fig
    # fig = plt.figure()
    # plt.imshow(F, cmap=cm.Greys_r)
    # fig.savefig(imgname)
    # plt.close(fig)
    # plt.close()

def write_image_multiple(op_images, imshape, pnt_nos, pathtmpl):
    # TODO: Fix this
    # pnt_nos = [0]
    for idx, p in enumerate(pnt_nos):
        if len(imshape) == 3:
            beg = idx * imshape[0] * imshape[1] * imshape[2]
            end = beg + imshape[0] * imshape[1] * imshape[2]
        else:
            beg = idx * imshape[0] * imshape[1]
            end = beg + imshape[0] * imshape[1]
        
        write_image(op_images[:, beg:end], imshape, pathtmpl.format(p)) 
        
def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful 
        eg. to display the weights of a neural network layer.
    """
    numimages = M.shape[1]
    if layout is None:
        n0 = int(np.ceil(np.sqrt(numimages)))
        n1 = int(np.ceil(np.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * np.ones(((height + border) * n0 + border, (width + border) * n1 + border), dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i * n1 + j < M.shape[1]:
                im[i * (height + border) + border:(i + 1) * (height + border) + border,
                   j * (width + border) + border :(j + 1) * (width + border) + border] = np.vstack((
                            np.hstack((np.reshape(M[:, i * n1 + j], (height, width)),
                                   bordercolor * np.ones((height, border), dtype=float))),
                            bordercolor * np.ones((border, width + border), dtype=float)
                            ))
    plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest', **kwargs)
    plt.show()

def lcn_2d(im, sigmas=[1.591, 1.591]):
    """ Apply local contrast normalization to a square image.
    Uses a scheme described in Pinto et al (2008)
    Based on matlab code by Koray Kavukcuoglu
    http://cs.nyu.edu/~koray/publis/code/randomc101.tar.gz

    data is 2-d
    sigmas is a 2-d vector of standard devs (to define local smoothing kernel)
    
    Example
    =======
    im_p = lcn_2d(im,[1.591, 1.591])
    """

    # assert(issubclass(im.dtype.type, np.floating))
    im = np.cast[np.float](im)

    # 1. subtract the mean and divide by std dev
    mn = np.mean(im)
    sd = np.std(im, ddof=1) 
    im -= mn
    if np.allclose(sd, 0.0):
        return im
    im /= sd

    # # 2. compute local mean and std
    # kerstring = '''0.0001    0.0005    0.0012    0.0022    0.0027    0.0022    0.0012    0.0005    0.0001
    #     0.0005    0.0018    0.0049    0.0088    0.0107    0.0088    0.0049    0.0018    0.0005
    #     0.0012    0.0049    0.0131    0.0236    0.0288    0.0236    0.0131    0.0049    0.0012
    #     0.0022    0.0088    0.0236    0.0427    0.0520    0.0427    0.0236    0.0088    0.0022
    #     0.0027    0.0107    0.0288    0.0520    0.0634    0.0520    0.0288    0.0107    0.0027
    #     0.0022    0.0088    0.0236    0.0427    0.0520    0.0427    0.0236    0.0088    0.0022
    #     0.0012    0.0049    0.0131    0.0236    0.0288    0.0236    0.0131    0.0049    0.0012
    #     0.0005    0.0018    0.0049    0.0088    0.0107    0.0088    0.0049    0.0018    0.0005
    #     0.0001    0.0005    0.0012    0.0022    0.0027    0.0022    0.0012    0.0005    0.0001'''
    # ker = []
    # for l in kerstring.split('\n'):
    #     ker.append(np.fromstring(l, dtype=np.float, sep=' '))
    # ker = np.asarray(ker)

    # lmn = scipy.signal.correlate2d(im, ker, mode='same', boundary='symm')
    # lmnsq = scipy.signal.correlate2d(im ** 2, ker, mode='same', boundary='symm')

    lmn = gaussian_filter(im, sigmas, mode='reflect')
    lmnsq = gaussian_filter(im ** 2, sigmas, mode='reflect')

    lvar = lmnsq - lmn ** 2;
    # lvar = np.where( lvar < 0, lvar, 0)
    np.clip(lvar, 0, np.inf, lvar)  # items < 0 set to 0
    lstd = np.sqrt(lvar)

    np.clip(lstd, 1, np.inf, lstd)

    im -= lmn
    im /= lstd

    return im


def sliding_window(im, win_height=128, win_width=64, step=1):
   """Returns a view win into im where win[i,j] is a view of the
i,j'th window in im."""

   H, W = im.shape[:2]
   nh = (H - win_height) / step + 1
   nw = (W - win_width) / step + 1

   # Get the original strides.
   strides = np.asarray(im.strides)

   # The first two strides also advance in the x,y directions
   new_strides = tuple(np.concatenate((strides[:2] * step, strides)))

   # The new shape, this should allow for grayscale/color images in
   # the final position.
   new_shape = tuple([nh, nw, win_height, win_width] + list(im.shape[2:]))

   # Create a view into the image array.
   windows = stride_tricks.as_strided(im, new_shape, new_strides)

   return windows

def gaussian2D(imshape, x_mean, y_mean, sigma):
    # imshape = (width, height)
    # Set up the 2D Gaussian:
    delta = 1
    x = np.arange(0, imshape[0], delta)
    y = np.arange(0, imshape[1], delta)
    X, Y = np.meshgrid(x, y)
    if not (x_mean <= 0 or x_mean >= imshape[0] or y_mean <= 0 or y_mean >= imshape[1]):
        Z = mlab.bivariate_normal(X, Y, sigma, sigma, x_mean, y_mean)
    else:
        Z = np.zeros((imshape[1], imshape[0]))
    return Z
    
def rgb2gray(rgb):
    r, g, b = np.rollaxis(rgb[..., :3], axis= -1)
    return 0.21 * r + 0.71 * g + 0.07 * b

def blend_mask(im, color, mask):
    
    mask = mask * (mask > 80)  # 120
    
    for i in range(0, 3):
        im[:, :, i] = np.multiply((255 - mask) / 255.0, im[:, :, i])
        
    for i in range(0, 3):
        im[:, :, i] = im[:, :, i] + np.multiply((mask / 255.0), color[i])
    
    return im

 
def run_command(command_line):
    print 'Exec: ' + command_line
    args = shlex.split(command_line)
    p = sp.Popen(args, stdout=sp.PIPE)
    while True:
        output = p.stdout.readline()
        if output == '' and p.poll() != None:
            break
        if output != '':   
            sys.stdout.write('\r' + output)       
            sys.stdout.flush()
            
def download_file_nyu(sourcepath, local_dir):
    
    if os.path.exists(local_dir + re.split(r'[\\/]+', sourcepath)[-1]):
        print 'File already exists'
        return
    
    print 'Please be patient, downloading big files from ajain@nyu. Progress will be displayed shortly.'
    command_line = 'rsync -r -v --progress -e ssh ajain@access.cims.nyu.edu:{0:s} {1:s}'.format(sourcepath, local_dir)    
    run_command(command_line)        

def unzip(zipfilepath, local_dir):   
    print 'Unzipiing files into ' + local_dir
    command_line = 'unzip ' + zipfilepath + ' -d ' + local_dir            
    run_command(command_line)

def get_2d_index(index, imshape):
    row = index / imshape[1]
    col = index % imshape[1]
    return row, col
        
def calc_score(opimages, test_set_y, params, names):
    score = 0
    thresh = 5  # min(params.imshape[0], params.imshape[1]) / 10.0
    for im_idx in range(0, opimages.shape[0]):  
	idx_data = np.argmax(opimages[im_idx, :])
        idx_op = np.argmax(test_set_y[im_idx, :])    
        
        img = plt.imread(names[im_idx])        
        fig = plt.figure()
        
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img, cmap=cm.Greys_r)
        lims = ax.axis()
        y_op, x_op = get_2d_index(idx_op, params.imshape)
        ax.scatter(x_op, y_op)
        
	y_data, x_data = get_2d_index(idx_data, params.imshape)
        ax.scatter(x_data, y_data, c='r')
        
	if math.fabs(x_data - x_op) < thresh and math.fabs(y_data - y_op) < thresh:
            score += 1
	    ax.set_title('Correct')
	else:
	    ax.set_title('Incorrect') 
        # ax = fig.add_subplot(1,2,2)
        fig.savefig(params.op_dir + '/{0:s}{1:04d}.png'.format('test', im_idx), bbox_inches=0)
        
        print 'score {0:d} / {1:d}'.format(score, im_idx)
    
    print 'final score: {0:d} / {1:d}'.format(score, opimages.shape[0])

def pickle_dump(var, filepath):
    f = open(filepath, 'w+')
    pickle.dump(var, f)
    f.close()
    print 'Successfully pickled to ' + filepath

def pickle_load(filepath):  
    if os.path.exists(filepath):
        print 'File exists, loading ' + filepath
        f = open(filepath, 'r+')
        var = pickle.load(f)
        f.close()
    else:
        var = None
        raise IOError
    return var


        
def pad_all_side_one_channel(im, pad_size, border):
    # create new image
    new_im = np.zeros((im.shape[0] + 2 * pad_size, im.shape[1] + 2 * pad_size), dtype=im.dtype)
    # place old image in the center
    new_im[pad_size:pad_size + im.shape[0], pad_size:pad_size + im.shape[1]] = im
    # pad top and left
    margin_top = 0
    margin_bot = 0
    if border == 'copy':
        margin_top = im[0, :]
        margin_bot = im[-1, :]
        
    new_im[0:pad_size, pad_size:pad_size + im.shape[1]] = margin_top
    new_im[pad_size + im.shape[0]:, pad_size:pad_size + im.shape[1]] = margin_bot

    margin_left = 0
    margin_right = 0
    if border == 'copy':
        margin_left = new_im.T[pad_size, :]
        margin_right = new_im.T[pad_size + im.shape[1] - 1]
    
    new_im.T[0:pad_size] = margin_left
    new_im.T[pad_size + im.shape[1]:] = margin_right
    return new_im

def pad_allsides(im, pad_size, border='copy'):
    nd = 1
    if len(im.shape) > 2:
        nd = im.shape[2]        

    if nd == 1:
        new_im = pad_all_side_one_channel(im, pad_size, type)
    else:
        new_im = np.zeros((im.shape[0] + 2 * pad_size, im.shape[1] + 2 * pad_size, 3), dtype=im.dtype)
        for i in range(0, nd):
            new_im[:, :, i] = pad_all_side_one_channel(im[:, :, i], pad_size, type)
    return new_im

def plot_scaled_pixel_prec(distances, params):
    # now plot distances
    name_map = ['Face', 'Shoulder', 'Elbow', 'Wrist']
    x = []
    y = []
    fig = plt.figure()
    for thresh in range(0, 40):
        # count how many less than thresh
        count_for_currthresh = 0
        for dist in distances:
            if dist < thresh:
                count_for_currthresh += 1
        x.append(thresh)
        y.append(count_for_currthresh)
    y = [(iy / 200.0) * 100 for iy in y]
    plt.plot(x, y)
    plt.title(name_map[params.pnt_nos[0]], fontsize=22)
    plt.xlabel('threshold in pixel', fontsize=16)
    plt.ylabel('% of examples', fontsize=16)
    fig.savefig(params.op_dir + '/plot.png')
    
def get_gpu_fit_size(X, already_alloc_mem=0):
    d_types = ['train', 'valid', 'test']
    gpu_size = dict() 
    for d_type in d_types:
        gpu_size[d_type] = X[d_type].shape[0] 
    if theano.theano.config.device.startswith('gpu'):  
        mem_requirements = [X[d_type].nbytes for d_type in d_types]
        total_mem_required = sum(mem_requirements)    
        free_mem, total_size = cuda.mem_info()
        free_mem += already_alloc_mem
        free_mem *= 0.9  # to be on the safe side, and to account for loading annotations which I am not counting for now
        if free_mem < total_mem_required:
            red_ratio = float(free_mem) / total_mem_required
            for d_type in d_types:
                gpu_size[d_type] = int(X[d_type].shape[0] * red_ratio)
            # gpu_size = {d_type:int(X[d_type].shape[0] * red_ratio) for d_type in d_types}
    return gpu_size

    
    
def gaussian_filter(kernel_shape):

    x = np.zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            x[i, j] = gauss(i - mid, j - mid)

    return x / np.sum(x)

def lecun_lcn(input, img_shape, kernel_shape, threshold=1e-4):
    """
    Yann LeCun's local contrast normalization
    Orginal code in Theano by: Guillaume Desjardins
    """
    input = input.reshape(input.shape[0], 1, img_shape[0], img_shape[1])
    X = T.matrix(dtype=theano.config.floatX)
    X = X.reshape(input.shape)

    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = gaussian_filter(kernel_shape).reshape(filter_shape)

    convout = conv.conv2d(input=X,
                             filters=filters,
                             image_shape=(input.shape[0], 1, img_shape[0], img_shape[1]),
                             filter_shape=filter_shape,
                             border_mode='full')

    # For each pixel, remove mean of 9x9 neighborhood
    
    mid = int(np.floor(kernel_shape / 2.))
    centered_X = X - convout[:, :, mid:-mid, mid:-mid]
    # Scale down norm of 9x9 patch if norm is bigger than 1
    sum_sqr_XX = conv.conv2d(input=centered_X ** 2,
                             filters=filters,
                             image_shape=(input.shape[0], 1, img_shape[0], img_shape[1]),
                             filter_shape=filter_shape,
                             border_mode='full')

    denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
    per_img_mean = denom.mean(axis=[1, 2])
    divisor = T.largest(per_img_mean.dimshuffle(0, 'x', 'x', 1), denom)
    divisor = T.maximum(divisor, threshold)

    new_X = centered_X / divisor
    new_X = new_X.dimshuffle(0, 2, 3, 1)
    new_X = new_X.flatten(ndim=3)

    f = theano.function([X], new_X)
    return f(input)

def flatten_windows(slid_window):
    flattened_wins = numpy.ndarray(shape=(slid_window.shape[0] * slid_window.shape[1], numpy.prod(slid_window.shape[2:])), dtype=theano.config.floatX)
    idx = 0
    for i0 in range(slid_window.shape[0]):
        for i1 in range(slid_window.shape[1]):
            win = slid_window[i0][i1]
            win = numpy.rot90(win)
            flattened_wins[idx] = win.T.flatten()
            idx += 1
    return flattened_wins

def bisect(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > a[mid]: hi = mid
        else: lo = mid + 1
    return lo

def dist_2D((x1, y1), (x2, y2)):
    dist_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return math.sqrt(dist_2)

def rot2D((x, y), deg, (center_x, center_y)=(0, 0)):
    rot_angle = math.radians(deg)
    xrot = center_x + (x - center_x) * math.cos(rot_angle) - (y - center_y) * math.sin(rot_angle)
    yrot = center_y + (x - center_x) * math.sin(rot_angle) + (y - center_y) * math.cos(rot_angle)
    return (xrot, yrot)

def parse_idl_file(filename):
    f = open(filename)
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    new_lines = []
    for line in lines:
        k1 = line.find('"')
        k2 = line.rfind('"')
        new_lines.append(line[k1 + 1:k2])
    lines = new_lines
    return lines

def report_progress(idx, total):
    per_complete = '{0}/{1}'.format(idx, total)
    sys.stdout.write("\r%s" % per_complete)
    sys.stdout.flush()
    
def set_sym_diff(set1, set2):
    set3 = union(set1 - set2, set2 - set1)
    return list(set3)

