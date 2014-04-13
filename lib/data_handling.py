from common_imports import *

import warnings
warnings.filterwarnings("ignore")

class Data(object):
    kinds = ['train', 'test']
    def __init__(self):
        self.X_names = defaultdict(list)
        self.X = defaultdict(list)
        self.Y = defaultdict(list)
        self.__jnt_pos_2d = defaultdict(list)
        
    def add_to_dataset(self, data_path):
        
        img_names_glob = glob.glob(data_path + '/*.jpg')
        img_names_glob += glob.glob(data_path + '/*.png')
        
        # Load images
        kind = data_path.split('/')[-2]
        assert kind in Data.kinds, 'kind is {0}'.format(kind)
        for img_name in img_names_glob:
            split_names = re.split(r'[\\/]+', img_name)
            x_name = split_names[-1]
            self.X_names[kind].append(data_path + '/' + x_name)
            
        # Load annotations
        self.__jnt_pos_2d[kind] = tools.pickle_load(data_path + '/jnt_pos_2d.pkl')
            
        if tools.run_on_subset == True:
            self.X_names[kind] = self.X_names[kind][0:5]
        
        print 'No of images added: ' + str(len(self.X_names[kind]))
            
    def save_picked_data(self, op_dir):
        if os.path.exists(op_dir + '/Data_part1.pkl'):
            return
        tools.pickle_dump([self.X_names, self.img0], op_dir + '/Data_part1.pkl')
        for kind in Data.kinds:
            numpy.save(op_dir + '/Data_part2_' + kind, self.X[kind])
            numpy.save(op_dir + '/Data_part3_' + kind, self.Y[kind])
    
    def load_picked_data(self, op_dir):
        print 'Looking for file: ' + op_dir + '/Data_part1.pkl'
        if (not os.path.exists(op_dir + '/Data_part1.pkl')):
            return False

        self.X_names, self.img0 = tools.pickle_load(op_dir + '/Data_part1.pkl')
                
        for kind in Data.kinds:
            self.X[kind] = numpy.load(op_dir + '/Data_part2_' + kind + '.npy')
            self.Y[kind] = numpy.load(op_dir + '/Data_part3_' + kind + '.npy')

        self.data_loaded = True
            
    def load_data(self, windowshape=(0, 0)):        
        if(self.data_loaded):
            return       
        # open first image for size    
        self.img0 = plt.imread(self.X_names['train'][0])
        self.img0 = tools.pad_allsides(self.img0, windowshape[0] / 2)
        self.no_channels = self.img0.shape[2] if len(self.img0.shape) > 2 else 1
        for kind in Data.kinds:
            self.X[kind] = numpy.ndarray(shape=(len(self.X_names[kind]), self.img0.shape[0], \
                                                self.img0.shape[1], self.no_channels), dtype=theano.config.floatX)
            self.Y[kind] = numpy.ndarray(shape=(len(self.X_names[kind]), len(self.__jnt_pos_2d['train'].values()[0]), 2), dtype=theano.config.floatX)
        # load x in memory   
        for kind in Data.kinds:
            print 'No of images loaded: ' 
            for idx in range(0, len(self.X_names[kind])):
                x_img = plt.imread(self.X_names[kind][idx]);
                x_img = tools.pad_allsides(x_img, windowshape[0] / 2)
                self.X[kind][idx] = x_img
                del x_img
                
                imname = self.X_names[kind][idx]
                jnt_key = re.sub('/+', '/', "/".join(imname.split('/')[2:]))
                for pntno, pnt in enumerate(self.__jnt_pos_2d[kind][jnt_key]): 
                    self.Y[kind][idx][pntno] = numpy.asarray([pnt[0] + windowshape[0] / 2, pnt[1] + windowshape[1] / 2]) 
                
                tools.report_progress(idx, len(self.X_names[kind]))        
            print ' '
            
    def preprocess(self):   
        if(self.data_loaded):
            return  
        print 'Preprocessing input images'
        for kind in Data.kinds:
            print '\t- ' + kind
            if self.X[kind].shape[0] > 0:
                for d in range(self.X[kind].shape[3]):
                    print '\t\t- channel ' + str(d)
                    # do in batches of max 1000 images each
                    im_idx = 0
                    while im_idx < self.X[kind].shape[0]:
                        step_size = min(1000, self.X[kind].shape[0] - im_idx)
                        print ' - Lecun GPU processing {0:d} images'.format(step_size)
                        # TODO: Note how '9' is fixed as kernel size for lcn
                        self.X[kind][im_idx:im_idx + step_size, :, :, d] = \
                                    tools.lecun_lcn(self.X[kind][im_idx:im_idx + step_size, :, :, d], self.img0.shape[0:2], 9)
                        im_idx += step_size
        
class DataSlidingWindow(Data):
    def __init__(self, points):
        super(DataSlidingWindow, self).__init__()   
        # indexes in images for +ve and -ve parts
        self.X_SW_p = defaultdict(list)
        self.X_SW_n = defaultdict(list)
        # Symmetric points on  right for the left parts
        self.pnt_map_sym = [0, 4, 5, 6]
        self.data_loaded = False
        # This is the isolation size for bounding box overlap
        self.isolate_size = 20
        # TODO: This is FLIC specific
        self.single_people_images = defaultdict(list)
        self.single_people_images['train'] = [1, 2, 3, 4, 5, 6, 42, 46, 47, 48, 49, 63, 64, 65, 66, 67, 77, 80, 98, 101, 107, 108, 109, 110, 123, 219, 220, 233, 234, 235, 237, 243, 244, 245, 248, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 425, 426, 427, 463, 464, 465, 466, 483, 484, 485, 486, 487, 488, 489, 490, 491, 493, 494, 495, 496, 497, 498, 499, 500, 509, 521, 538, 550, 552, 553, 554, 555, 556, 557, 561, 562, 563, 564, 565, 566, 575, 576, 577, 578, 579, 580, 583, 584, 587, 588, 589, 590, 591, 592, 593, 594, 595, 657, 659, 660, 661, 662, 663, 664, 678, 679, 680, 789, 790, 791, 825, 827, 857, 936, 937, 938, 939, 940, 941, 945, 946, 947, 948, 949, 950, 953, 954, 959, 969, 970, 971, 972, 973, 974, 975, 976, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1198, 1199, 1200, 1201, 1202, 1203, 1210, 1213, 1214, 1215, 1216, 1217, 1220, 1221, 1225, 1226, 1227, 1249, 1250, 1251, 1252, 1315, 1316, 1317, 1318, 1361, 1367, 1368, 1375, 1376, 1382, 1383, 1384, 1385, 1386, 1387, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1421, 1422, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1590, 1591, 1603, 1605, 1606, 1607, 1616, 1617, 1618, 1623, 1627, 1628, 1630, 1635, 1636, 1637, 1638, 1639, 1640, 1650, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1815, 1816, 1825, 1876, 1877, 1878, 1879, 1888, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1916, 1917, 1918, 1919, 1922, 1929, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1961, 1965, 1968, 2013, 2014, 2015, 2016, 2017, 2018, 2024, 2025, 2026, 2032, 2033, 2034, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2054, 2055, 2056, 2057, 2058, 2059, 2083, 2107, 2108, 2109, 2112, 2125, 2126, 2127, 2128, 2136, 2211, 2212, 2227, 2381, 2388, 2390, 2448, 2455, 2456, 2457, 2458, 2459, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2486, 2487, 2488, 2489, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2546, 2644, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2689, 2690, 2694, 2695, 2696, 2697, 2703, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2846, 2847, 2848, 2849, 2859, 2860, 2861, 2862, 2863, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2878, 2879, 2880, 2881, 2882, 2886, 2889, 2890, 2894, 2895, 2896, 2897, 2907, 2908, 2909, 2910, 2911, 2912, 2920, 2990, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3090, 3091, 3097, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3147, 3149, 3156, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3240, 3241, 3242, 3315, 3316, 3318, 3319, 3320, 3321, 3322, 3324, 3327, 3328, 3350, 3351, 3352, 3353, 3354, 3370, 3371, 3372, 3386, 3387, 3401, 3402, 3437, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3488, 3489, 3490, 3492, 3493, 3497, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3520, 3521, 3566, 3630, 3631, 3632, 3633, 3634, 3668, 3669, 3670, 3671, 3680, 3681, 3682, 3683, 3684, 3693, 3694, 3695, 3696, 3697, 3698, 3701, 3708, 3709, 3710, 3711, 3712, 3713, 3725, 3736, 3737, 3738, 3741, 3745, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3798, 3799, 3800, 3802, 3808, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3827, 3828, 3829, 3830, 3831, 3843, 3858, 3863, 3865, 3866, 3873, 3887, 3888, 3889, 3890, 3891, 3892, 3898, 3921, 3922, 3923, 3924, 3925, 3937, 3938, 3968, 3969, 3970, 3971, 3972, 4006, 4007, 4008, 4009, 4010, 4021, 4055, 4056, 4057, 4059, 4081, 4083, 4084, 4085, 4086, 4093, 4201, 4202, 4203, 4204, 4205, 4206, 4238, 4239, 4273, 4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281, 4283, 4284, 4285, 4286, 4287, 4296, 4297, 4298, 4299, 4300, 4301, 4310, 4311, 4312, 4313, 4314, 4315, 4316, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4453, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 4535, 4546, 4547, 4548, 4586, 4587, 4588, 4589, 4590, 4591, 4592, 4593, 4594, 4595, 4596, 4597, 4599, 4600, 4601, 4602, 4603, 4604, 4605, 4606, 4607, 4608, 4609, 4610, 4611, 4612, 4613, 4614, 4615, 4616, 4617, 4618, 4619, 4620, 4622, 4627, 4650, 4651, 4652, 4653, 4654, 4655, 4656, 4657, 4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669, 4676, 4677, 4678, 4687, 4688, 4689, 4690, 4691, 4703, 4823, 4824, 4874, 4900, 4901, 4902, 4903, 4904, 4905, 4910, 4933, 4934, 4935, 4936, 4937, 4942, 4943, 4944, 4945, 4946, 4947, 4948, 4950, 4951, 4954, 4955, 4956, 4961, 4962, 4963, 4964, 4965]
        # this defines which part (face, wrist, etc)
        self.curr_pnt = points[0]
        
    def save_picked_data(self, op_dir):
        super(DataSlidingWindow, self).save_picked_data(op_dir)   
        if os.path.exists(op_dir + '/Data_part6.pkl'):
            return
        tools.pickle_dump([self.window_size, self.pnt_map_sym], op_dir + '/Data_part6.pkl')
        for kind in Data.kinds:
            numpy.save(op_dir + '/Data_part4_' + kind, self.X_SW_p[kind])
            numpy.save(op_dir + '/Data_part5_' + kind, self.X_SW_n[kind])
        
    def load_picked_data(self, op_dir):
        super(DataSlidingWindow, self).load_picked_data(op_dir)   
        print 'Looking for file: ' + op_dir + '/Data_part6.pkl'
        if (not os.path.exists(op_dir + '/Data_part6.pkl')):
            return False
        self.window_size, self.pnt_map_sym = tools.pickle_load(op_dir + '/Data_part6.pkl')
                
        for kind in Data.kinds:
            self.X_SW_p[kind] = numpy.load(op_dir + '/Data_part4_' + kind + '.npy')
            self.X_SW_n[kind] = numpy.load(op_dir + '/Data_part5_' + kind + '.npy')

        self.data_loaded = True

    
    def load_data(self, windowshape, shuffle=False):    
        if(self.data_loaded):
            return
        super(DataSlidingWindow, self).load_data(windowshape)        
        super(DataSlidingWindow, self).preprocess()
        
        self.window_size = (windowshape[0], windowshape[1], self.no_channels)
        
        # Load window-locations in images for +ves
        for kind in Data.kinds:
            for idx in range(0, len(self.X_names[kind])):
                part_pos = self.Y[kind][idx][self.curr_pnt]
                is_valid = self.get_part_win_positive(part_pos)
                if not is_valid:
                    print 'Couldnt get +ve win in image: ' + self.X_names[kind][idx]
                else:
                    self.X_SW_p[kind].append([idx, part_pos[0], part_pos[1]])
        
        # Load window-locations in images for -ves by sampling
        sampling_stride = 16
        for kind in Data.kinds:
            print '\nCreating "negative" {0} by keeping bounding box and sampling'.format(kind)
            for idx in range(0, len(self.X[kind])):
                tools.report_progress(idx, len(self.X[kind]))
                for x0 in range(0, self.img0.shape[1], sampling_stride):
                    for y0 in range(0, self.img0.shape[0], sampling_stride):
                        is_valid = self.get_part_win_negative(idx, [x0, y0], self.Y[kind][idx])
                        if is_valid:
                            self.X_SW_n[kind].append([idx, x0, y0])   
        
        for kind in Data.kinds:
            random.shuffle(self.X_SW_n[kind])
            random.shuffle(self.X_SW_p[kind])
        
    

    def get_part_win_negative(self, idx, win_pos, all_joints):
        pos = all_joints[self.curr_pnt]
        pos_sym = all_joints[self.pnt_map_sym[self.curr_pnt]]
        bbox_pad = self.window_size[0] / 2
        x = win_pos[0]
        y = win_pos[1]

        min_x = 0
        min_y = 0
        max_x = self.img0.shape[1]
        max_y = self.img0.shape[0]
        if idx not in self.single_people_images['train']:
            max_x = max(all_joints[:, 0]) + bbox_pad
            max_y = max(all_joints[:, 1]) + bbox_pad
            
            max_x = min(self.img0.shape[1], max_x)
            max_y = min(self.img0.shape[0], max_y)
            
            min_x = min(all_joints[:, 0]) - bbox_pad
            min_y = min(all_joints[:, 1]) - bbox_pad
            
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            
        is_neg_win_valid = False
        if x - self.window_size[0] / 2 >= min_x  and y - self.window_size[1] / 2 >= min_y \
            and x + self.window_size[0] / 2 < max_x  and y + self.window_size[0] / 2 < max_y:
            is_neg_win_valid = True
        return is_neg_win_valid
     
    def get_part_win_positive(self, pos):
        imshape = (self.img0.shape[0], self.img0.shape[1], 3)  
        offset = self.window_size[0] / 2
        is_valid = False
        if pos[1] - offset >= 0 and  pos[0] - offset >= 0 and \
                        pos[0] + offset < imshape[1] and pos[1] + offset < imshape[0]:
            is_valid = True
        return is_valid

            


