import numpy
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage.filters as filters 


op_hmap_path_prefix = 'V:/ajain/Projects/deep_nets/tmp/'

# Load scale 5 for size, and bring every image to this size
filename = op_hmap_path_prefix + '/' + 'beforesp_sc{0}.npy'.format(5)
scale_5 = numpy.load(filename)

# now load every other and bring it to this scale
composite = scale_5
for scale in [0, 1, 2, 3, 4]:
    # load
    filename = op_hmap_path_prefix + '/' + 'beforesp_sc{0}.npy'.format(scale)
    hmap = numpy.load(filename)
    # scale
    xfac = scale_5.shape[0] / float(hmap.shape[0]) 
    yfac = scale_5.shape[1] / float(hmap.shape[1]) 
    hmap = scipy.ndimage.interpolation.zoom(hmap, (xfac, yfac))
    # keep the biggest
    composite[composite < hmap] = hmap[composite < hmap]

plt.imshow(composite)
plt.show()
