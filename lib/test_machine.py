from common_imports import *

# import warnings
# warnings.filterwarnings("ignore")

if tools.compute_test_value:
    theano.config.compute_test_value = 'warn'

class TestMachine(Machine):
    n_false_pos = 0
    n_false_neg = 0
    
    def __init__(self, params):
        print 'Start'
        self.img_data = None    
        params.scale_to_run = 1.0
        params.max_imgs = None
        params.i0 = None
        params.test_idl_file = None
        params.st_frame = None
        params.end_frame = None
        
        
        self.layers = []
        #TODO
        self.step = 1 
        super(TestMachine, self).__init__(params)
      
    def __create_bprop_machine(self, params):
        # x values
        self.x = T.matrix('x')
    
        input = self.x
        idx = 0
        for layer_xml in params.conf.findall('layers/layer'):
            class_name = layer_xml.find('type').text
            prev_layer = self.layers[idx - 1] if idx > 0 else None
            layer = Machine.layer_map[class_name](layer_xml, params, prev_layer)   
            self.layers.append(layer)
            idx += 1  
            print '** Created {0:s} {1:s}'.format(layer.type, layer.id)   
            
        for layer in self.layers:
            layer.compute(input, params)
            input = layer.output
            print '++ Computed {0:s} {1:s}'.format(layer.type, layer.id)   
        
    def __create_fprop_machine(self, params):        
        train_batch_size = params.batch_size
        params.batch_size = 1
        idx = 0
        input = self.x
        for layer_xml in params.conf.findall('layers/layer'):
            class_name = layer_xml.find('type').text
            prev_layer = self.layers[idx - 1] if idx > 0 else None
            if class_name.startswith('Input') or class_name.startswith('Conv'):
                if class_name.startswith('Input'):
                    layer = self.layer_map[class_name](layer_xml, params, prev_layer, self.imshape)   
                else:
                    layer = self.layer_map[class_name](layer_xml, params, prev_layer)  
                    self.step *= layer.pool_size 
                self.layers[idx] = layer
                idx += 1  
                print '** Created {0:s} {1:s}'.format(layer.type, layer.id)   
            
        for layer in self.layers:
            class_name = layer.type
            if class_name.startswith('Input') or class_name.startswith('Conv'):
                layer.compute(input, params)
                input = layer.output
                print '++ Computed {0:s} {1:s}'.format(layer.type, layer.id)
        params.batch_size = train_batch_size
        # now, create new 'input' for the output layers and beyond
        self.conv_out_gpu = T.tensor4(name='convout', dtype=theano.config.floatX) 
        input = self.conv_out_gpu
        for layer in self.layers:
            class_name = layer.type
            if not (class_name.startswith('Input') or class_name.startswith('Conv')):
                layer.compute(input, params)
                input = layer.output
                print '++ Computed {0:s} {1:s}'.format(layer.type, layer.id)

    def evaluate_image(self, imname, params):
        conv_out = self.get_output_conv()
        window_height = self.window_size[0]/float(self.step)
        print 'Window {0} --> {1} after pooling'.format(self.window_size[0], window_height)
        
        tmp = numpy.zeros((conv_out.shape[2], conv_out.shape[3]), dtype=theano.config.floatX)
        tmp = tools.sliding_window(tmp, window_height, window_height)
        
        sw_r = 0
        sw_c = 0
        self.no_windows = numpy.prod(tmp.shape[:2])
        mul_of_batchsize = self.no_windows - self.no_windows % params.batch_size
        op_density = numpy.zeros(shape=(self.no_windows), dtype=theano.config.floatX)
        batch_size = params.batch_size
        for b_no in range(0, (mul_of_batchsize / batch_size) - 1):  # no of batches that can fit the slidinging windows
            print b_no
            # tools.report_progress(100 * (b_no + 1), float(mul_of_batchsize / batch_size - 1))
            # set one batch
            for j0 in range(0, params.batch_size):
                self.conv_out_data[j0, :, :, :] = conv_out[0, :, sw_r:sw_r + window_height, sw_c:sw_c + window_height]
                sw_c += 1
                if sw_c + window_height == conv_out.shape[3] + 1:
                    sw_c = 0
                    sw_r += 1
                    
            self.conv_out_data_gpu.set_value(self.conv_out_data)
            
            op = numpy.asarray([self.get_output_sw_op()]).T.ravel()
            conf = numpy.asarray([self.get_output_sw_conf()]).T.ravel()
            zeros = numpy.arange(op.shape[0])[op == tools.NEGLABEL]
            ones = numpy.arange(op.shape[0])[op == tools.POSLABEL]
            assert op.shape[0] == ones.shape[0] + zeros.shape[0]
            op_density[b_no * batch_size: (b_no + 1) * batch_size][ones] = conf[ones]
            op_density[b_no * batch_size: (b_no + 1) * batch_size][zeros] = 1.0 - conf[zeros]
            
            # TODO: Count false positives
            # TODO: Count false negatives
            # TODO: Make an image of false positives and false negatives
                
        # tools.write_image(op_density, (1,1), params.op_dir + 'slidwinop_{0:d}.png'.format(i0))
        op_density = op_density.reshape((tmp.shape[0], tmp.shape[1]))
        op_density = numpy.fliplr(op_density)
        neighborhood_size = 5
        data_max = filters.maximum_filter(op_density, neighborhood_size)
        maxima = (op_density == data_max)
        op_density_maxima = numpy.copy(op_density)
        op_density_maxima[maxima == False] = 0
        
        op_density_maxima[op_density_maxima < .1] = 0
        rows, cols = numpy.nonzero(op_density_maxima)
        jnt_pos_2d = dict()
        jnt_pos_2d[imname] = []
        xs = []
        ys = []
        ss = [] 
        for idx in range(0, rows.shape[0]):
            y = rows[idx] 
            x = cols[idx] 
            score = op_density_maxima[y, x]
            x = x * self.step / params.scale_to_run
            y = y * self.step / params.scale_to_run
            print 'x:{0:f}, y:{1:f}'.format(x, y)
            jnt_pos_2d[imname].append((x, y, score)) 
            xs.append(x)
            ys.append(y)
            ss.append(score)
        scorelist = jnt_pos_2d[imname]
        jnt_pos_2d[imname] = sorted(scorelist, key=lambda x: x[2], reverse=True)
        al_write_dir = params.op_dir + '/' + imname.split('/')[-2] + '_sc_{0}_ep_{1}'.format(params.scale_to_run, params.epoch_no)

        if not os.path.exists(al_write_dir):
            os.makedirs(al_write_dir)
        al.write_many(jnt_pos_2d, al_write_dir + '/' + re.split(r'[\\/]+', imname)[-1].split('.')[0] + '.al')
        fig = plt.figure() 
        im = plt.imread(imname)
        plt.imshow(im)
        plt.scatter(xs, ys, c=ss, cmap=cm.coolwarm)
        if imname in jnt_pos_2d and len(jnt_pos_2d[imname]) > 0 and  len(jnt_pos_2d[imname][0]) > 0:
            plt.scatter(jnt_pos_2d[imname][0][0], jnt_pos_2d[imname][0][1], c='g')
        fig.savefig(al_write_dir + '/' + re.split(r'[\\/]+', imname)[-1])
        
        fig = plt.figure() 
        plt.imshow(op_density)
        fig.savefig(al_write_dir + '/' + 'op_den_' + re.split(r'[\\/]+', imname)[-1])
        
        numpy.save(al_write_dir + '/' + re.split(r'[\\/]+', imname)[-1].split('.')[0], op_density)
        
    def compute(self, params):
        ###############
        # TEST MODEL #
        ###############
        print '... testing'
        if params.test_idl_file != None:
            img_names_glob = tools.parse_idl_file(params.test_idl_file)
            print 'Using test idl file: ' + params.test_idl_file
        else:
            img_names_glob = glob.glob(params.test_dir + '/*.png')
            img_names_glob += glob.glob(params.test_dir + '/*.jpg')
            print 'Test dir: ' + params.test_dir
        #######################
        # BProp
        #######################
        self.__create_bprop_machine(params)   
        #######################
        i_input = 0
        assert self.layers[i_input].type == "InputLayerSW"
        
        self.window_size = self.layers[i_input].windowshape
        # Note, assuming all images to be of the same size        
        x_img = plt.imread(img_names_glob[0])
        x_img = scipy.ndimage.interpolation.zoom(x_img, (params.scale_to_run, params.scale_to_run, 1))
        x_img = tools.pad_allsides(x_img, self.window_size[0]/2.0)
        x_img = numpy.asarray(x_img, dtype=theano.config.floatX)
        self.imshape = x_img.shape
        x_img = numpy.rot90(x_img)
        x_img = x_img.T.flatten()
        x_img = x_img.reshape((1, x_img.shape[0]))
        self.img_data = theano.shared(x_img, borrow=True)
        #######################
        # FProp
        #######################
        self.__create_fprop_machine(params)
        #######################
        # Find the last ConvPoolLayer
        i_conv_last = -1
        for i in range(len(self.layers)):
            if (self.layers[i].type == "ConvPoolLayer"):
                i_conv_last = i
        assert i_conv_last != -1
        
        self.get_output_conv = theano.function([], self.layers[i_conv_last].output,
                        givens={self.x: self.img_data})
        
        window_height = self.window_size[0]/float(self.step)
        self.conv_out_data = numpy.zeros((params.batch_size, self.layers[i_conv_last].nkerns, window_height, window_height), dtype=theano.config.floatX)
        self.conv_out_data_gpu = theano.shared(self.conv_out_data)
        
        self.get_output_sw_op = theano.function([], self.layers[-1].output,
                    givens={self.conv_out_gpu: self.conv_out_data_gpu})
        if self.layers[-1].type == 'SoftMaxLayer':
            self.get_output_sw_conf = theano.function([], self.layers[-1].confidence,
                    givens={self.conv_out_gpu: self.conv_out_data_gpu})
        
        if params.max_imgs != None:
            img_names_glob = img_names_glob[0:params.max_imgs]
            
        if tools.run_on_subset:
            img_names_glob = img_names_glob[0:2]
        
        # Turn off all dropout layers (which turn into a static multiplier
        # layer, with m = 1 - prop) during testing.
        DropoutLayer.SetDropoutOn(False)
        
        st_frame = 0
        end_frame = len(img_names_glob)
        
        n_false_pos = 0
        n_false_neg = 0
        
        if params.st_frame != None:
            assert params.end_frame != None
            assert params.i0 == None
            st_frame = params.st_frame
            end_frame = params.end_frame
            
        if params.i0 != None:    
            st_frame = params.i0
            end_frame = st_frame+1
        for i0 in range(st_frame, end_frame):
            imname = img_names_glob[i0]
            # load image and lcn preprocess
            x_img = plt.imread(imname) 
            x_img = scipy.ndimage.interpolation.zoom(x_img, (params.scale_to_run, params.scale_to_run, 1))
            x_img = tools.pad_allsides(x_img, self.window_size[0] / 2.0)
            x_img = x_img.reshape(1, x_img.shape[0], x_img.shape[1], x_img.shape[2]) 
            for d in range(3):
                x_img[:, :, :, d] = tools.lecun_lcn(x_img[:, :, :, d], (x_img.shape[1], x_img.shape[2]), 9)
            x_img = x_img[0]
            
            # bring imgae to right format
            x_img = numpy.rot90(x_img)
            x_img = x_img.T.flatten()
            x_img = x_img.reshape((1, x_img.shape[0]))
    
            self.img_data.set_value(x_img, borrow=True)

            self.evaluate_image(imname, params)
