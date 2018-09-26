"""Script for training and testing
Author: Guotai Wang
"""
import os
import sys
import math
import random
import nibabel
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.data import Iterator
from niftynet.layer.loss_segmentation import LossFunction as SegmentationLoss
from Demic.net.net_factory import NetFactory
from Demic.image_io.file_read_write import save_array_as_nifty_volume
from Demic.image_io.image_loader import ImageLoader
from Demic.util.image_process import *
from Demic.train_test.test_func import volume_probability_prediction_3d_roi,convert_label

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label 
        input_tensor: tensor with shae [N, D, H, W, 1]
        output_tensor: shape [N, D, H, W, num_class]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = tf.equal(input_tensor, i*tf.ones_like(input_tensor,tf.int32))
        tensor_list.append(temp_prob)
    output_tensor = tf.concat(tensor_list, axis=-1)
    output_tensor = tf.cast(output_tensor, tf.float32)
    return output_tensor

def soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    pred   = tf.reshape(prediction, [-1, num_class])
    pred   = tf.nn.softmax(pred)
    ground = tf.reshape(soft_ground_truth, [-1, num_class])
    n_voxels = ground.get_shape()[0].value
    if(weight_map is not None):
        weight_map = tf.reshape(weight_map, [-1])
        weight_map_nclass = tf.reshape(
            tf.tile(weight_map, [num_class]), pred.get_shape())
        ref_vol = tf.reduce_sum(weight_map_nclass*ground, 0)
        intersect = tf.reduce_sum(weight_map_nclass*ground*pred, 0)
        seg_vol = tf.reduce_sum(weight_map_nclass*pred, 0)
    else:
        ref_vol = tf.reduce_sum(ground, 0)
        intersect = tf.reduce_sum(ground*pred, 0)
        seg_vol = tf.reduce_sum(pred, 0)
    dice_numerator = 2*tf.reduce_sum(intersect)
    dice_denominator = tf.reduce_sum(seg_vol + ref_vol)
    dice_score = dice_numerator/dice_denominator
    return 1-dice_score

class TrainTestAgent(object):
    def __init__(self, config, mode = 'train'):
        self.mode = mode
        self.config_data = config['data']
        self.config_net  = config['network']
        self.net_params  = config['network_parameter']
        if(mode == 'train'):
            self.config_train= config['training']
            self.batch_size  = self.config_train.get('batch_size', 5)
            seed = self.config_train.get('random_seed', 1)
            tf.set_random_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        else:
            self.config_test = config['testing']
            self.batch_size  = self.config_test.get('batch_size', 5)

    def _construct_network(self):
        # 1, get input and output shape
        self.full_data_shape = [self.batch_size] + self.config_net['patch_shape_x']
        self.full_out_shape  = [self.batch_size] + self.config_net['patch_shape_y']

        self.x = tf.placeholder(tf.float32, shape = self.full_data_shape)
        self.w = tf.placeholder(tf.float32, shape = self.full_out_shape)
        self.y = tf.placeholder(tf.int32, shape = self.full_out_shape)
        self.m = tf.placeholder(tf.float32, shape = []) # momentum for batch normalization
        
        # 2, define network
        if(self.mode == 'train'):
            w_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
            b_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        else:
            w_regularizer = None
            b_regularizer = None
        if(type(self.config_net['net_type']) is str):
            net_class = NetFactory.create(self.config_net['net_type'])
        else:
            print('customized network is used')
            net_class = self.config_net['net_type']
        self.class_num = self.config_net['class_num']
        self.net = net_class(num_classes = self.class_num,
                        parameters  = self.net_params,
                        w_regularizer = w_regularizer,
                        b_regularizer = b_regularizer,
                        name = self.config_net['net_name'])
        self.predicty = self.net(self.x, is_training = self.mode == 'train', bn_momentum=self.m)
        print('network output shape ', self.predicty.shape)
        if(self.mode == 'train'):
            self.__set_loss_and_optimizer()
    
    def __set_loss_and_optimizer(self):
        # 1, set loss function
        loss_type = self.config_train.get('loss_type', 'Dice')
        loss_func = SegmentationLoss(self.class_num, loss_type)
        self.loss = loss_func(self.predicty, self.y, weight_map = self.w)
        
        # 2, set optimizer
        lr = self.config_train.get('learning_rate', 1e-3)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch normalization
        with tf.control_dependencies(update_ops):
            self.opt_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
            
    def _load_data(self):
        self.config_data['batch_size'] = self.batch_size
        self.config_data['patch_type_x'] = self.config_net['patch_type_x']
        self.config_data['patch_type_y'] = self.config_net['patch_type_y']
        self.config_data['patch_shape_x'] = self.config_net['patch_shape_x']
        self.config_data['patch_shape_y'] = self.config_net['patch_shape_y']
        if(self.mode == 'train'):
            # Place data loading and preprocessing on the cpu
            with tf.device('/cpu:0'):
                self.data_agent = ImageLoader(self.config_data)
                # create an reinitializable iterator given the dataset structure
                train_dataset = self.data_agent.get_dataset('train')
                valid_dataset = self.data_agent.get_dataset('valid')
                train_iterator = Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)
                valid_iterator = Iterator.from_structure(valid_dataset.output_types,
                                                   valid_dataset.output_shapes)
                self.next_train_batch = train_iterator.get_next()
                self.next_valid_batch = valid_iterator.get_next()
            # Ops for initializing the two different iterators
            self.train_init_op = train_iterator.make_initializer(train_dataset)
            self.valid_init_op = valid_iterator.make_initializer(valid_dataset)
        else:
            self.data_agent = ImageLoader(self.config_data)
            self.test_dataset = self.data_agent.get_dataset('test')


    def train(self):
        # 1, construct network and create data generator
        self._construct_network()
        self._load_data()
        
        # 2, start the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        max_epoch   = self.config_train['maximal_epoch']
        loss_file   = self.config_train['model_save_prefix'] + "_loss.txt"
        start_epoch = self.config_train.get('start_epoch', 0)
        loss_list   = []
        if( start_epoch> 0):
            saver.restore(self.sess, self.config_train['pretrained_model'])
        
        for epoch in range(start_epoch, max_epoch):
            # 3, Initialize iterators and train for one epoch
            temp_momentum = float(epoch)/float(max_epoch)
            train_loss_list = []
            self.sess.run(self.train_init_op)
            for step in range(self.config_train['batch_number']):
                one_batch = self.sess.run(self.next_train_batch)
                feed_dict = {self.x:one_batch['image'],
                             self.w:one_batch['weight'],
                             self.y:one_batch['label'],
                             self.m:temp_momentum}
                self.opt_step.run(session = self.sess, feed_dict=feed_dict)
                if(step < self.config_train['test_steps']):
                    loss_train = self.loss.eval(feed_dict)
                    train_loss_list.append(loss_train)
            batch_loss = np.asarray(train_loss_list, np.float32).mean()
            epoch_loss = [batch_loss]
            
            if(self.config_data.get('data_names_val', None) is not None):
                valid_loss_list = []
                self.sess.run(self.valid_init_op)
                for test_step in range(self.config_train['test_steps']):
                    one_batch = self.sess.run(self.next_valid_batch)
                    feed_dict = {self.x:one_batch['image'],
                                 self.w:one_batch['weight'],
                                 self.y:one_batch['label'],
                                 self.m:temp_momentum}
                    loss_valid = self.loss.eval(feed_dict)
                    valid_loss_list.append(loss_valid)
                batch_loss = np.asarray(valid_loss_list, np.float32).mean()
                epoch_loss.append(batch_loss)
                
            print("{0:} Epoch {1:}, loss {2:}".format(datetime.now(), epoch+1, epoch_loss))
            
            # 4, save loss and snapshot
            loss_list.append(epoch_loss)
            np.savetxt(loss_file, np.asarray(loss_list))
            if((epoch+1)%self.config_train['snapshot_epoch']  == 0):
                saver.save(self.sess, self.config_train['model_save_prefix']+"_{0:}.ckpt".format(epoch+1))

    def __test_one_volume(self, img):
        # 1, caulculate shape of tensors
        batch_size = self.config_test.get('batch_size', 1)
        data_shape = self.config_net['patch_shape_x']
        label_shape= self.config_net['patch_shape_y']
        class_num  = self.config_net['class_num']
        margin     = [data_shape[i] - label_shape[i] for i in range(len(data_shape))]
    
        # 2, get test mode.
        #    0, (default) use defined tensor shape and original image shape
        #    1, use defined tensor shape, and resize image in 2D to match tensor shape
        #    2, use original image shape, and resize tensor in 2D to match image shape
        #    3, use original image shape, and resize tensor in 3D to match image shape
        shape_mode = self.config_test.get('shape_mode', 1)
        if(shape_mode == 1):
            [D0, H0, W0, C0] = img.shape
            resized_shape = [D0, data_shape[1], data_shape[2], C0]
            resized_img = resize_ND_volume_to_given_shape(img, resized_shape, order = 3)
        else:
            resized_img = img
        [D, H, W, C] = resized_img.shape
        
        # 3, pad input image to desired size when the network requires the input should be
        #    a multiple of size_factor, e.g. 16
        size_factor = self.config_test.get('size_factor',[1,1,1])
        Dr = int(math.ceil(float(D)/size_factor[0])*size_factor[0])
        Hr = int(math.ceil(float(H)/size_factor[1])*size_factor[1])
        Wr = int(math.ceil(float(W)/size_factor[2])*size_factor[2])
        pad_img = np.random.normal(size = [Dr, Hr, Wr, C])
        pad_img[np.ix_(range(D), range(H), range(W), range(C))] = resized_img
        
        if(shape_mode==3):
            data_shape[0] = Dr
            label_shape[0]= Dr - margin[0]
        data_shape[1] = Hr
        data_shape[2] = Wr
        label_shape[1]= Hr - margin[1]
        label_shape[2]= Wr - margin[2]
        full_data_shape  = [batch_size] + data_shape
        full_label_shape = [batch_size] + label_shape
        full_weight_shape = [i for i in full_data_shape]
        full_weight_shape[-1] = 1
        
        # 4, construct graph
        x = tf.placeholder(tf.float32, shape = full_data_shape)
        predicty = self.net(x, is_training = False, bn_momentum = 1.0)
        print('network input  ', x)
        print('network output ', predicty)
        proby = tf.nn.softmax(predicty)
        
        # 3, load model
#        self.sess = tf.InteractiveSession()
#        self.sess.run(tf.global_variables_initializer())
#        saver = tf.train.Saver()
#        saver.restore(self.sess, self.config_test['model_file'])

        # 5, inference
        outputp = volume_probability_prediction_3d_roi(pad_img, data_shape, label_shape,
                                                       class_num, batch_size, self.sess, x, proby)
        outputy = np.asarray(np.argmax(outputp, axis = 3), np.uint16)
        outputy = outputy[np.ix_(range(D), range(H), range(W))]
        if(shape_mode == 1):
            outputy = resize_ND_volume_to_given_shape(outputy, img.shape[:-1], order = 0)
        return outputy

    def test(self):
        # 1, load data
        self._construct_network()
        self._load_data()
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, self.config_test['model_file'])
        
#        return
        # 2, test each data
        label_source = self.config_data.get('label_convert_source', None)
        label_target = self.config_data.get('label_convert_target', None)
        if(not(label_source is None) and not(label_source is None)):
            assert(len(label_source) == len(label_target))

        for one_data in self.test_dataset:
            img  = one_data['image']
            name = one_data['name']
            print(img.shape, name)
            pred = self.__test_one_volume(img)
            if (label_source is not None and label_target is not None):
                pred = convert_label(pred, label_source, label_target)
            save_name = '{0:}_{1:}.nii.gz'.format(name, self.config_data['output_postfix'])
            save_array_as_nifty_volume(pred, self.config_data['save_root']+'/'+save_name)


