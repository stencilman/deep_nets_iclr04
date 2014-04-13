from common_imports import *
import threading
import Queue

# import warnings
# warnings.filterwarnings("ignore")

if tools.compute_test_value:
    theano.config.compute_test_value = 'warn'
    

class TrainMachine(Machine):
    training_finished = False  # When this is true the preturb thread will quit
    
    def __init__(self, params):  
        super(TrainMachine, self).__init__(params)
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        # x values
        self.x = T.matrix('x')
        # y values
        self.y = T.vector('y', dtype=theano.config.floatX)
        
        self.__create_bprop_machine(params)   
                       
        if self.layers[-1].type == 'SoftMaxLayer':
            self.cost = -T.mean(T.log(self.layers[-1].p_y_given_x)[T.arange(self.y.shape[0]), T.cast(self.y, 'int32')])
            self.cost_sans_reg = -T.mean(T.log(self.layers[-1].p_y_given_x)[T.arange(self.y.shape[0]), T.cast(self.y, 'int32')])
        else:
            self.cost = self.layers[-1].svm_cost(self.y)  # T.mean(T.nnet.binary_crossentropy(self.layers[-1].output.flatten(), self.y)) #((self.y - self.layers[-1].output.flatten()) ** 2).sum()            

    	print 'Regularization: ' + str(params.reg_weight)
        
    	if not numpy.allclose(params.reg_weight, 0.0):
                for layer in self.layers:
                    if hasattr(layer, 'W'):
                        self.cost += params.reg_weight * (layer.W.val ** 2).sum()
                        # TODO: Try regularization of the bias values as well
                        
        # create a list of all model parameters to be fit by gradient descent
        self.model_params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                self.model_params += layer.params
        # velocities to create momentum 
        self.velocities = [theano.shared(value=numpy.zeros_like(p.get_value(), \
                                    dtype=p.dtype), \
                                    borrow=True) for p in self.model_params]
        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.model_params)
    
        # create the updates list by automatically looping over all
        # (params[i],grads[i]) and the rmsprop updates if we need them
        self.updates = []
        
        # If we're performing RMSPROP the first update is calculation of the
        # new running average of the current sq gradient. Hinton's Lecture 6e: 
        # MS(w,t) = 0.9*MS(w,t-1) + 0.1*(de/dw)^2
        # the ms value is therefore per parameter (so is kinda expensive)
        if params.use_rmsprop:
            print 'Using RMSPROP'
            print 'rmsprop_filter_weight: ' + str(params.rmsprop_filter_weight)
            print 'rmsprop_maxgain: ' + str(params.rmsprop_maxgain)
            print 'rmsprop_mingain: ' + str(params.rmsprop_mingain)
            # ms is the running sum of the gradient squared (for each weight)
            self.ms = [theano.shared(value=numpy.ones_like(p.get_value(), \
                                     dtype=p.dtype), \
                                     borrow=True) for p in self.model_params]
            fw = params.rmsprop_filter_weight
            ms_min = params.rmsprop_mingain
            ms_max = params.rmsprop_maxgain
            
            for grad_i, ms_i in zip(self.grads, self.ms):
                upd = fw * ms_i + (1 - fw) * (grad_i ** 2)
                self.updates.append((ms_i, T.clip(upd, ms_min, ms_max)))
            
        else:
            print 'Not using RMSPROP'
            self.ms = None
            
        if params.use_momentum:
            mom = params.momentum
            print 'Using Momentum'
            print 'momentum: ' + str(params.momentum)
        else:
            print 'Not using Momentum'

        # Make learning rate a shared variable, so that we can potentially
        # adjust it dynamically later.
        print 'learning_rate: ' + str(params.learning_rate) 
        self.lr = T.shared(numpy.cast[theano.config.floatX](params.learning_rate), \
                           name="ms", borrow=True)
        for i in range(0, len(self.model_params)):
            param_i = self.model_params[i]
            grad_i = self.grads[i]
            vel_i = self.velocities[i]
            
            effective_lr = None
            if params.use_rmsprop:
                ms_i = self.ms[i]
                effective_lr = self.lr / T.sqrt(ms_i)
            else:
                effective_lr = self.lr
            if params.use_momentum:
                # TODO: Change to Nesterov momentum.
                upd = mom * vel_i + (1 - mom) * grad_i
                self.updates.append((vel_i, upd))
                self.updates.append((param_i, param_i - effective_lr * upd))
            else:
                self.updates.append((param_i, param_i - effective_lr * grad_i))
        
        self.__define_theano_functions(params)
        print 'Finished defining model parameter updates'
        if os.path.exists(params.op_dir + '/weak_neg.pkl'):
            tools.pickle_load(params.op_dir + '/weak_neg.pkl')
        # To make sure it is filled once in the beginning
        print 'First time filling, to make sure it is filled once in the beginning'
        self.filling_data = threading.Lock()
        self.avail_data_queue = Queue.Queue()
        # Start thread for start filling for next epoch
        print 'Started new thread for next round of filling'
        fill_thread = threading.Thread(target=self.__fill_new_data, args=(params,))
        fill_thread.start()
        
    def __create_data_containers(self, params):
        ##########################################################################################
        # Create data containers which keep getting transfered to the gpu
        ##########################################################################################
        gpu_batches = tools.no_batches_on_gpu * params.batch_size
        self.no_images_gpu = dict()
        for kind in ['train', 'test']:
            self.no_images_gpu[kind] = min(gpu_batches, len(self.layers[0].data.X_SW_p[kind]))
            print "self.no_images_gpu[{0:s}]: {1:d}".format(kind, self.no_images_gpu[kind])    
        
        print '----> Positive : Negatives = {0} : {1}'.format(1, params.mix_ratio)
        winsize = numpy.prod(params.imshape)
        containter_size_multiplier = {'train':1 + 1.0 / params.mix_ratio['train'], 'test':1 + 1.0 / params.mix_ratio['test']}
        self.train_x = numpy.zeros((self.no_images_gpu['train'] * containter_size_multiplier['train'], winsize), dtype=theano.config.floatX)
        self.train_y = numpy.zeros((self.no_images_gpu['train'] * containter_size_multiplier['train'], 1), dtype=theano.config.floatX)
        self.test_x = numpy.zeros((self.no_images_gpu['test'] * containter_size_multiplier['test'], winsize), dtype=theano.config.floatX)
        self.test_y = numpy.zeros((self.no_images_gpu['test'] * containter_size_multiplier['test'], 1), dtype=theano.config.floatX)
        
        # structrues for negative mining
        self.no_of_neg_patches_per_transfer_in_hard_mine = params.batch_size * 200
        self.neg_batch_on_cpu = numpy.zeros((self.no_of_neg_patches_per_transfer_in_hard_mine, winsize), dtype=theano.config.floatX)
        self.neg_batch_on_gpu = theano.shared(self.neg_batch_on_cpu, borrow=True)
                
        self.theano_x = dict()
        self.theano_y = dict()
        
        self.idx_pos = {'train':0, 'test':0}
        self.idx_neg = {'train':0, 'test':0}
        self.weak_negatives = set()
        
        self.fill_count = 0
        
        self.theano_x['train'] = theano.shared(self.train_x, borrow=True)
        self.theano_x['test'] = theano.shared(self.test_x, borrow=True)
        self.theano_y['train'] = theano.shared(self.train_y, borrow=True)
        self.theano_y['test'] = theano.shared(self.test_y, borrow=True)
        
        if tools.compute_test_value:
            self.index.tag.test_value = 0
            self.y.tag.test_value = self.train_y.flatten()[0:params.batch_size]
            self.x.tag.test_value = self.train_x[0:params.batch_size]
    
    @staticmethod
    def __rotate(x_img, (xpos, ypos), offset, magnitude):
        imshape = x_img.shape
        extra_pad_fac = 1.5
        if xpos - extra_pad_fac * offset >= 0 and  ypos - extra_pad_fac * offset >= 0 and \
                        xpos + extra_pad_fac * offset < imshape[1] and ypos + extra_pad_fac * offset < imshape[0]:
            double_win = x_img[ypos - extra_pad_fac * offset: ypos + extra_pad_fac * offset, xpos - extra_pad_fac * offset: xpos + extra_pad_fac * offset]
            double_win = scipy.ndimage.rotate(double_win, magnitude, order=1)
            # print 'Rotate Success'
            return double_win
            
        else:
            return x_img[ypos - offset: ypos + offset, xpos - offset: xpos + offset]
    
    @staticmethod
    def  __translate(x_img, (xpos, ypos), offset, t_x, t_y):
        imshape = x_img.shape
        if xpos - t_x - offset >= 0 and  ypos - t_y - offset >= 0 and \
                        xpos - t_x + offset < imshape[1] and ypos - t_x + offset < imshape[0]:
            # print 'Translate Success'
            return True
        else:
            return False
    
    @staticmethod
    def  __scale(x_img, (xpos, ypos), offset, magnitude):
        working_img = scipy.ndimage.interpolation.zoom(x_img, (magnitude, magnitude, 1), order=1)
        imshape = working_img.shape
        if xpos - offset >= 0 and  ypos - offset >= 0 and \
                        xpos + offset < imshape[1] and ypos + offset < imshape[0]:
            win = working_img[ypos - offset: ypos + offset, xpos - offset: xpos + offset]
            # print 'Scale Success'
            return win
        else:
            return x_img[ypos - offset: ypos + offset, xpos - offset: xpos + offset]
    
    @staticmethod
    def __get_window(x_img, (xpos, ypos), offset, perturb):
        if not perturb:
            return x_img[ypos - offset: ypos + offset, xpos - offset: xpos + offset]
        
        # Rotate between +/- [0, 25] deg
        magnitude = random.uniform(0, 20) * random.choice([1, -1])
        double_win = TrainMachine.__rotate(x_img, (xpos, ypos), offset, magnitude)
        # Translate between 0, 10 px
        t_x = random.uniform(0, 10) * random.choice([1, -1])
        t_y = random.uniform(0, 10) * random.choice([1, -1])
        translate_is_possible = TrainMachine.__translate(double_win, (double_win.shape[1] / 2, double_win.shape[0] / 2), offset, t_x, t_y)
        if not translate_is_possible:
            t_x = 0
            t_y = 0
        # Scale between 80%-120%
        magnitude = random.uniform(.8, 1.2) 
        final_win = TrainMachine.__scale(double_win, (double_win.shape[1] / 2 - t_x, double_win.shape[0] / 2 - t_y), offset, magnitude)
        
        return final_win
            
    def __fill_new_data(self, params):
        while not self.training_finished:
            print 'In fill data no: ' + str(self.fill_count)
            self.fill_count += 1
            
            x = {'train':self.train_x, 'test':self.test_x}
            y = {'train':self.train_y, 'test':self.test_y}
            offset = params.imshape[0] / 2
            
            for kind in ['train', 'test']:
                idx_in_batch = 0
                negatives_to_add = 0
                
                # Fill positve
                start_time = time.clock()
                for i in range (self.no_images_gpu[kind]):
                    # tools.report_progress(i, self.no_images_gpu[kind])
                    posidx = self.idx_pos[kind]
                    idx, xpos, ypos = self.layers[0].data.X_SW_p[kind][posidx]
                    poswin = TrainMachine.__get_window(self.layers[0].data.X[kind][idx], (int(xpos), int(ypos)), offset, params.perturb)
                    x[kind][idx_in_batch, :] = numpy.rot90(poswin).T.flatten()
                    y[kind][idx_in_batch, :] = tools.POSLABEL
                    idx_in_batch += 1
                    self.idx_pos[kind] += 1
                    if self.idx_pos[kind] == len(self.layers[0].data.X_SW_p[kind]):
                        self.idx_pos[kind] = 0
                    # For one positive we have 1.0 / params.mix_ratio negatives
                    negatives_to_add += 1.0 / params.mix_ratio[kind]
                end_time = time.clock()
                print >> sys.stderr, ('Time to fill +ves %.5fs' % ((end_time - start_time)))
            
                # Fill negative 
                start_time = time.clock()
                for i in range (int(negatives_to_add)):                
                    negidx = self.idx_neg[kind]
                    while kind == 'train' and negidx in self.weak_negatives:
                        self.idx_neg[kind] += 1
                        negidx = self.idx_neg[kind]
                    
                    idx, xpos, ypos = self.layers[0].data.X_SW_n[kind][negidx]
                    negwin = TrainMachine.__get_window(self.layers[0].data.X[kind][idx], (int(xpos), int(ypos)), offset, params.perturb)
                    x[kind][idx_in_batch, :] = numpy.rot90(negwin).T.flatten()
                    y[kind][idx_in_batch, :] = tools.NEGLABEL
                    idx_in_batch += 1
                    self.idx_neg[kind] += 1
                    
                    if self.idx_neg[kind] == len(self.layers[0].data.X_SW_n[kind]):
                        self.idx_neg[kind] = 0
                        print 'Exhausted NEG for {0:s}, rewinding'.format(kind)
                end_time = time.clock()
                print >> sys.stderr, ('Time to fill -ves %.5fs' % ((end_time - start_time)))
    
    #         kind = 'test'
    #         tools.write_image(x[kind], params.imshape, params.op_dir + 'X-{0:s}-{1:d}.png'.format(kind, self.fill_count))
    #         tools.write_image_multiple(y[kind], (1, 1), params.pnt_nos, params.op_dir + 'Y-{0:s}-{1:d}'.format(kind, self.fill_count) + '{0:d}.png')
            print 'Waiting for epoch to finish'
            self.filling_data.acquire()
            for kind in ['train', 'test']:
                indexes = range(len(x[kind]))
                random.shuffle(indexes)
                shuff_x = x[kind][indexes]
                shuff_y = y[kind][indexes]
                self.theano_x[kind].set_value(shuff_x, borrow=True)
                self.theano_y[kind].set_value(shuff_y, borrow=True)
            self.avail_data_queue.put(True)
            self.filling_data.release()
        
    
    def mine_negatives(self, params):
        print 'Mining hard negatives'
        print 'Start weak neg : ' + str(len(self.weak_negatives))
        print 'Checking how current model is doing on previously collected weak negatives'
        mine_on = self.weak_negatives
        weak_negatives = self.mine_negatives_data(params, mine_on)   
        print '\n{0} not weak anymore. Removing them from weak negatives'.format(len(mine_on - weak_negatives))
        self.weak_negatives = self.weak_negatives - (mine_on - weak_negatives)
        
        print '\nCollecting more weak negatives:'
        print 'Total weak neg before: ' + str(len(self.weak_negatives))
        mine_on = set(range(len(self.layers[0].data.X_SW_n['train'])))
        mine_on = set.union(mine_on - self.weak_negatives, self.weak_negatives - mine_on) 
        weak_negatives = self.mine_negatives_data(params, mine_on)
        self.weak_negatives = set.union(self.weak_negatives, weak_negatives)
        print 'Total weak neg after: ' + str(len(self.weak_negatives))
        
        
    def mine_negatives_data(self, params, mine_on):   
        if len(mine_on) == 0:
            return mine_on    
        negidx = 0
        mine_thresh = 0.99
        kind = 'train'
        weak_negatives = set()
        no_done = 1
        while 1:
            tools.report_progress(no_done, len(mine_on))
            neg_idx_one_batch = []
            conf_one_transfer = []
            for idx_in_batch in range(self.no_of_neg_patches_per_transfer_in_hard_mine):
                if negidx > max(mine_on):
                    return weak_negatives
                neg_idx_one_batch.append(negidx)
                offset = params.imshape[0] / 2
                idx, xpos, ypos = self.layers[0].data.X_SW_n[kind][negidx]
                negidx += 1
                no_done += 1
                negwin = self.layers[0].data.X[kind][idx][int(ypos) - offset: int(ypos) + offset, int(xpos) - offset: int(xpos) + offset]
                self.neg_batch_on_cpu[idx_in_batch, :] = numpy.rot90(negwin).T.flatten()
            # Done building the batch, transfer to GPU and calculate the confidence of patches in this batch
            self.neg_batch_on_gpu.set_value(self.neg_batch_on_cpu, borrow=True)
            # Calucate the confidences of these negatives
            for minibatch_index in xrange(self.no_of_neg_patches_per_transfer_in_hard_mine / params.batch_size):
                [conf, op_labels] = self.layers[-1].get_output_layer_vals_mine(minibatch_index)                
                ones = numpy.arange(conf.shape[0])[op_labels == tools.POSLABEL]
                conf = numpy.asarray(conf, dtype=theano.config.floatX)
                conf[ones] = 1.0 - conf[ones]                
                conf_one_transfer += conf.tolist()
            assert len(conf_one_transfer) == len(neg_idx_one_batch)
            weak_neg = set()
            for idx in range(len(conf_one_transfer)):
                if conf_one_transfer[idx] > mine_thresh:
                    weak_neg.add(neg_idx_one_batch[idx])
            weak_negatives = set.union(weak_negatives, weak_neg)  
        
    def __define_theano_functions(self, params):
        
        
        for layer in self.layers[1:]:
            layer.get_output_layer = dict()
            for kind in ['train', 'test']:
                layer.get_output_layer[kind] = theano.function([self.index], layer.output,
                                                                givens={self.x: self.theano_x[kind][self.index * params.batch_size: 
                                                                (self.index + 1) * params.batch_size]})
        
        self.layers[-1].get_output_layer_vals_mine = theano.function([self.index], [layer.confidence, layer.output],
                                                                givens={self.x: self.neg_batch_on_gpu[self.index * params.batch_size: 
                                                                (self.index + 1) * params.batch_size]})
     
        # actual theano functions used during training
        
        self.train_model = theano.function([self.index], [self.cost, self.cost_sans_reg], updates=self.updates,
                 givens={
                    self.x: self.theano_x['train'][self.index * params.batch_size: (self.index + 1) * params.batch_size],
                    self.y: self.theano_y['train'].flatten()[self.index * params.batch_size: (self.index + 1) * params.batch_size]})
                
        self.test_model = theano.function([self.index], [self.cost, self.cost_sans_reg],
                 givens={
                    self.x: self.theano_x['test'][self.index * params.batch_size: (self.index + 1) * params.batch_size],
                    self.y: self.theano_y['test'].flatten()[self.index * params.batch_size: (self.index + 1) * params.batch_size]})
            
        self.get_grads = theano.function([self.index], self.grads,
                givens={
                self.x: self.theano_x['train'][self.index * params.batch_size: (self.index + 1) * params.batch_size],
                self.y: self.theano_y['train'].flatten()[self.index * params.batch_size: (self.index + 1) * params.batch_size]})
    
    def __create_bprop_machine(self, params):
        
        print '... building the model'
        
        self.layers = []
        input = self.x
        idx = 0
        for layer_xml in params.conf.findall('layers/layer'):
            class_name = layer_xml.find('type').text
            prev_layer = self.layers[idx - 1] if idx > 0 else None
            layer = Machine.layer_map[class_name](layer_xml, params, prev_layer)   
            
            # Make sure the layer's output and input sizes were defined properly
            assert layer.out_size != (0)
            assert layer.in_size != (0)
            
            self.layers.append(layer)
            if class_name.startswith('Input'):
                self.__create_data_containers(params)
            idx += 1  
            print '** Created {0:s} {1:s}'.format(layer.type, layer.id)   
            
        for layer in self.layers:
            layer.compute(input, params)
            input = layer.output
            print '++ Computed {0:s} {1:s}'.format(layer.type, layer.id)   
        
        # write out the loaded data    
        if self.layers[0].log and not params.load_weights:
            self.layers[0].write_log(params)
            print 'Written Init Log'
            
    def compute(self, params):
        if tools.compute_test_value:  
            print >> sys.stderr, 'WARNING: compute_test_value = True (debug)'
        
        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        print '    hard_mine_freq = %d' % params.hard_mine_freq
        print '    n_iters = %d' % params.n_iters
        print '    n_epochs = %d' % params.n_epochs
        # early-stopping parameters
        patience = 30000000  # look as this many examples regardless
        epoch = params.epoch_no 
        iter = 0
        done_looping = False
        self.losses = defaultdict(list)
        
        while (iter < params.n_iters) and (not done_looping) and epoch < params.n_epochs:
            epoch = epoch + 1   
            print 'Waiting for new data.'
            is_data_avail = self.avail_data_queue.get()
            while not is_data_avail:
                is_data_avail = self.avail_data_queue.get()
                
            self.filling_data.acquire()
            print '\nNew data available.'
            epoch_start_time = time.clock()
            for minibatch_index in xrange(self.no_images_gpu['train'] / params.batch_size):
                # tools.report_progress(minibatch_index, self.no_images_gpu['train'] / params.batch_size)
                iter = (epoch - 1) * tools.no_batches_on_gpu + minibatch_index
                #************* COST ******************
                DropoutLayer.SetDropoutOn(True)
                [cost_ij, cost_sans_reg_ij] = self.train_model(minibatch_index)                    
                #*************************************
                if minibatch_index == self.no_images_gpu['train'] / params.batch_size - 1 and (epoch % 5 == 0 or epoch == 1):
                    DropoutLayer.SetDropoutOn(False)
                    
                    print 'training @ iter = ', iter
                    # Train set losses
                    train_losses = [self.train_model(i)[1] for i in xrange(self.no_images_gpu['train'] / params.batch_size)]
                    train_score = numpy.mean(train_losses)
                    self.losses['train'].append(train_score)
                    
                    # Test set losses
                    test_losses = [self.test_model(i)[1] for i in xrange(self.no_images_gpu['test'] / params.batch_size)]
                    test_score = numpy.mean(test_losses)
                    self.losses['test'].append(test_score)
    
                    self.__log_intermediate_info(params, epoch, iter)
                                        
                    print 'epoch {0:d}, minibatch {1:d}/{2:d}'.format(epoch, minibatch_index + 1, self.no_images_gpu['train'] / params.batch_size)
                   
                    print'         mean test error {0:f}'.format(test_score)
                    print'         mean train error {0:f}'.format(train_score)    
                            
            if epoch % params.hard_mine_freq == 0:
                DropoutLayer.SetDropoutOn(False)
                self.mine_negatives(params)
            self.filling_data.release()

            if epoch % 5 == 0:
                for layer in self.layers:
                    if hasattr(layer, 'W'):
                        layer.W.save_weight(params.weights_dir, epoch)
                    if hasattr(layer, 'b'):
                        layer.b.save_weight(params.weights_dir, epoch)
                tools.pickle_dump(self.weak_negatives, params.op_dir + '/weak_neg.pkl')
                
                if patience <= iter:
                    done_looping = True
                    break
            # self.__log_intermediate_info(params, epoch, 0)
            epoch_end_time = time.clock()
            print >> sys.stderr, ('Epoch time %.2fm' % ((epoch_end_time - epoch_start_time) / 60.))
   
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.W.save_weight(params.weights_dir, epoch)
            if hasattr(layer, 'b'):
                layer.b.save_weight(params.weights_dir, epoch)
        tools.pickle_dump(self.weak_negatives, params.op_dir + '/weak_neg.pkl')
       
        # Shut down the preturb thread
        self.training_finished = True

    def __log_intermediate_info(self, params, epoch, iter):
        paramNo = 0
        minibatch_index = 0
        curr_grads = self.get_grads(minibatch_index)                
        gradMag = numpy.float64(numpy.linalg.norm(curr_grads[paramNo]))  # float64 coz of numpy bug
        paramMag = numpy.float64(numpy.linalg.norm(self.model_params[paramNo].get_value()))  # float64 coz of numpy bug
        
        for idx, layer in enumerate(self.layers[1:]):
            if layer.log == True:
                layer.write_log(params, layer.get_output_layer['test'], epoch, iter)
            
        print 'Param[{0:d}] \n\tGrad magnitude:  = {1:f}\n\tParam Mag {2:f} '.format(paramNo, gradMag, paramMag)
        
        fig = plt.figure()
        for kind in ['train', 'test']:
            plt.plot(self.losses[kind])
        fig.savefig(params.op_dir + '/losses.png')
        plt.close(fig)
        
        # Print no of windows classified correctly
        data_x = {'train':self.train_x, 'test':self.test_x}
        data_y = {'train':self.train_y, 'test':self.test_y}
        
        for kind in ['train', 'test']:
            no_eg = int(data_x[kind].shape[0] / params.batch_size) * params.batch_size
            all_labels = numpy.zeros((no_eg))
            for minibatch_index in range(1, data_x[kind].shape[0] / params.batch_size):
                op_labels = self.layers[-1].get_output_layer[kind](minibatch_index)
                all_labels[minibatch_index * params.batch_size: 
                           (minibatch_index + 1) * params.batch_size] = op_labels.ravel()
            
            no_eg = all_labels.shape[0]
            index = range(0, no_eg)
            index = numpy.asarray(index)
            neg_eg = index[data_y[kind][0:no_eg].flatten() == tools.NEGLABEL]
            pos_eg = index[data_y[kind][0:no_eg].flatten() == tools.POSLABEL]
            
            no_pos_right = all_labels[pos_eg].sum()
            no_neg_right = len(neg_eg) - all_labels[neg_eg].sum()
            
            print ''
            print kind + ' positives : {0:f} / {1:f}'.format(no_pos_right, self.no_images_gpu[kind])
            print kind + ' negatives : {0:f} / {1:f}'.format(no_neg_right, self.no_images_gpu[kind] * params.mix_ratio[kind])
            
        

              
