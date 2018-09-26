"""Script for training
Author: Guotai Wang
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction as SegmentationLoss
from niftynet.layer.loss_regression import LossFunction as RegressionLoss
from Demic.util.parse_config import parse_config
from Demic.image_io.data_generator import ImageDataGenerator
from Demic.net.net_factory import NetFactory


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
    dice_score = 2.0*intersect/(ref_vol + seg_vol + 1.0)
    dice_score = tf.reduce_mean(dice_score)
    return 1.0-dice_score


def soft_cross_entropy_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    pred   = tf.reshape(prediction, [-1, num_class])
    pred   = tf.nn.softmax(pred)
    ground = tf.reshape(soft_ground_truth, [-1, num_class])
    ce = ground* tf.log(pred)
    if(weight_map is not None):
        n_voxels = tf.reduce_sum(weight_map)
        weight_map = tf.reshape(weight_map, [-1])
        weight_map_nclass = tf.reshape(
            tf.tile(weight_map, [num_class]), pred.get_shape())
        ce = ce * weight_map_nclass
    ce = -tf.reduce_mean(ce)
    return ce

def soft_size_loss1(prediction, soft_ground_truth, num_class, weight_map = None):
    pred = tf.reshape(prediction, [-1, num_class])
    pred = tf.nn.softmax(pred)
    grnd = tf.reshape(soft_ground_truth, [-1, num_class])

    pred_size = tf.reduce_sum(pred, 0)
    grnd_size = tf.reduce_sum(grnd, 0)
    dice_numerator   = 2*pred_size*grnd_size
    dice_denominator = tf.square(pred_size) + tf.square(grnd_size) + 1e-10
    size_loss = tf.div(dice_numerator, dice_denominator)
    size_loss = tf.reduce_sum(size_loss)
    size_loss = tf.div(size_loss, num_class)
    return 1-size_loss

def soft_size_loss(prediction, soft_ground_truth, num_class, weight_map = None):
    pred = tf.reshape(prediction, [-1, num_class])
    pred = tf.nn.softmax(pred)
    grnd = tf.reshape(soft_ground_truth, [-1, num_class])

    pred_size = tf.reduce_mean(pred, 0)
    grnd_size = tf.reduce_mean(grnd, 0)
    size_loss = tf.square(pred_size - grnd_size)
    size_loss = tf.multiply(size_loss, tf.constant([0] + [1]*(num_class-1), tf.float32))
    size_loss = tf.reduce_sum(size_loss)/(num_class - 1)
#    size_loss = size_loss/(tf.square(pred_size) + tf.square(grnd_size))
#    size_loss = tf.reduce_mean(size_loss)
    return size_loss

def soft_size_loss2(prediction, soft_ground_truth, num_class, weight_map = None):
    pred   = tf.reshape(prediction, [-1, num_class])
    pred   = tf.nn.softmax(pred)
    ground = tf.reshape(soft_ground_truth, [-1, num_class])
    pred_size   = tf.reduce_sum(pred, 0)
    ground_size = tf.reduce_sum(ground, 0)
    size_loss   = tf.square(pred_size - ground_size)
    size_loss   = tf.reduce_sum(size_loss)
    n = tf.shape(pred)
    n = tf.reduce_prod(n)
    size_loss = tf.div(size_loss, tf.cast(n, tf.float32))
    return size_loss

def get_loss_weights(iter, max_iter):
    N1 = int(max_iter/3)
    N2 = int(max_iter*2/3)
    if(iter < N1):
        lambda1 = 0.0
        lambda2 = 0.0
        lambda3 = 1.0
    elif(iter < N2):
        lambda1 = 0.33
        lambda2 = 0.33
        lambda3 = 0.33
    else:
        lambda1 = 1.0
        lambda2 = 0.0
        lambda3 = 0.0
    return [lambda1, lambda2, lambda3]

class TrainAgent(object):
    def __init__(self, config):
        self.config_data    = config['dataset']
        self.config_sampler = config['sampler']
        self.config_net     = config['network']
        self.net_params     = config['network_parameter']
        self.config_train   = config['training']
        
        seed = self.config_train.get('random_seed', 1)
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    def get_output_and_loss(self):
        pass

    def get_input_output_feed_dict(self):
        pass
    
    def get_loss(self, predict, ground_truth, class_num, weight_map = None):
        loss_name = self.config_train.get('loss_type', 'Dice')
        if(loss_name == 'CE'):
            return soft_cross_entropy_loss(predict, ground_truth, class_num, weight_map)
        else:
            return soft_dice_loss(predict, ground_truth, class_num, weight_map)

    def construct_network(self):
        batch_size  = self.config_sampler.get('batch_size', 5)
        self.full_data_shape = [batch_size] + self.config_sampler['data_shape']
        self.full_out_shape  = [batch_size] + self.config_sampler['label_shape']

        self.x = tf.placeholder(tf.float32, shape = self.full_data_shape)
        self.m = tf.placeholder(tf.float32, shape = []) # momentum for batch normalization
        self.get_output_and_loss()

    def get_variable_list(self, var_names, include = True):
        all_vars = tf.global_variables()
        output_vars = []
        for var in all_vars:
            if(include == False):
                output_flag = True
                if(var_names is not None):
                    for ignore_name in var_names:
                        if(ignore_name in var.name):
                            output_flag = False
                            break
            else:
                output_flag = False
                for include_name in var_names:
                    if(include_name in var.name):
                        output_flag = True
                        break
            if(output_flag):
                output_vars.append(var)
        return output_vars
        
    def create_optimization_step_and_data_generator(self):
        learn_rate  = self.config_train.get('learning_rate', 1e-3)
        vars_fixed  = self.config_train.get('vars_not_update', None)
        vars_update = self.get_variable_list(vars_fixed, include = False)
        update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch normalization
        with tf.control_dependencies(update_ops):
            self.opt_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss, var_list = vars_update)
        
        # Place data loading and preprocessing on the cpu
        with tf.device('/cpu:0'):
            self.train_data = ImageDataGenerator(self.config_data['data_train'], self.config_sampler)
            # create an reinitializable iterator given the dataset structure
            train_iterator = Iterator.from_structure(self.train_data.data.output_types,
                                               self.train_data.data.output_shapes)
            self.next_train_batch = train_iterator.get_next()
        # Ops for initializing the two different iterators
        self.train_init_op = train_iterator.make_initializer(self.train_data.data)
        valid_data       = []
        next_valid_batch = []
        valid_init_op    = []
        for i in range(len(self.config_data) - 1):
            with tf.device('/cpu:0'):
                temp_valid_data = ImageDataGenerator(self.config_data["data_valid{0:}".format(i)], self.config_sampler)
                temp_valid_iterator = Iterator.from_structure(temp_valid_data.data.output_types,
                                            temp_valid_data.data.output_shapes)
                temp_next_valid_batch = temp_valid_iterator.get_next()
            temp_valid_init_op = temp_valid_iterator.make_initializer(temp_valid_data.data)
            valid_data.append(temp_valid_data)
            next_valid_batch.append(temp_next_valid_batch)
            valid_init_op.append(temp_valid_init_op)
        self.valid_data = valid_data
        self.next_valid_batch = next_valid_batch
        self.valid_init_op = valid_init_op
        
    def train(self):
        # start the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
#        self.sess.run(tf.global_variables_initializer())
        save_vars = self.get_variable_list([self.config_net['net_name']], include = True)
        saver = tf.train.Saver(save_vars)
        
        max_iter    = self.config_train['maximal_iter']
        loss_file   = self.config_train['model_save_prefix'] + "_loss.txt"
        dice_file   = self.config_train['model_save_prefix'] + "_dice.txt"
        multi_scale_loss = self.config_train.get('multi_scale_loss', False)
        gradual_train    = self.config_train.get('gradual_train', False)
        start_iter  = self.config_train.get('start_iter', 0)
        loss_list, dice_list   = [], []
        if( start_iter > 0):
            vars_not_load = self.config_train.get('vars_not_load', None)
            restore_vars  = save_vars # self.get_variable_list(vars_not_load, include = False)
            restore_saver = tf.train.Saver(restore_vars)
            restore_saver.restore(self.sess, self.config_train['pretrained_model'])
        
        # make sure the graph is fixed during training
        tf.get_default_graph().finalize()
        self.sess.run(self.train_init_op)
        for valid_init_op in self.valid_init_op:
            self.sess.run(valid_init_op)
        
        for iter in range(start_iter, max_iter):
            # Initialize iterator with the training dataset
            temp_momentum = float(iter-start_iter)/float(max_iter-start_iter)
            try:
                feed_dict = self.get_input_output_feed_dict('train')
                feed_dict[self.m] = temp_momentum
                if(multi_scale_loss and gradual_train):
                    [lambda1, lambda2, lambda3] = get_loss_weights(iter, max_iter)
                    feed_dict[self.lambda1] = lambda1
                    feed_dict[self.lambda2] = lambda2
                    feed_dict[self.lambda3] = lambda3
                self.opt_step.run(session = self.sess, feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                self.sess.run(self.training_init_op)
            
            if((iter + 1) % self.config_train['test_interval'] == 0):
                batch_loss_list, batch_dice_list = [], []
                for test_step in range(self.config_train['test_steps']):
                    step_loss_list, step_dice_list = [], []
                    feed_dict = self.get_input_output_feed_dict('train')
                    feed_dict[self.m] = temp_momentum
                    if(multi_scale_loss and gradual_train):
                        [lambda1, lambda2, lambda3] = get_loss_weights(iter, max_iter)
                        feed_dict[self.lambda1] = lambda1
                        feed_dict[self.lambda2] = lambda2
                        feed_dict[self.lambda3] = lambda3
                    loss_v = self.loss.eval(feed_dict)
                    dice_v = self.dice.eval(feed_dict)
                    step_loss_list.append(loss_v)
                    step_dice_list.append(dice_v)
                    for valid_idx in range(len(self.config_data) - 1):
                        feed_dict = self.get_input_output_feed_dict('valid', valid_idx)
                        feed_dict[self.m] = temp_momentum
                        if(multi_scale_loss and gradual_train):
                            [lambda1, lambda2, lambda3] = get_loss_weights(iter, max_iter)
                            feed_dict[self.lambda1] = lambda1
                            feed_dict[self.lambda2] = lambda2
                            feed_dict[self.lambda3] = lambda3
                        loss_v = self.loss.eval(feed_dict)
                        dice_v = self.dice.eval(feed_dict)
                        step_loss_list.append(loss_v)
                        step_dice_list.append(dice_v)
                    batch_loss_list.append(step_loss_list)
                    batch_dice_list.append(step_dice_list)
                batch_loss = np.asarray(batch_loss_list, np.float32).mean(axis = 0)
                batch_dice = np.asarray(batch_dice_list, np.float32).mean(axis = 0)
                print("{0:} Iter {1:}, loss {2:}, dice {3:}".format(datetime.now(), \
                    iter+1, batch_loss, batch_dice))
                # save loss and snapshot
                loss_list.append(batch_loss)
                dice_list.append(batch_dice)
                np.savetxt(loss_file, np.asarray(loss_list))
                np.savetxt(dice_file, np.asarray(dice_list))
            if((iter+1)%self.config_train['snapshot_iter']  == 0):
                saver.save(self.sess, self.config_train['model_save_prefix']+"_{0:}.ckpt".format(iter+1))

class SegmentationTrainAgent(TrainAgent):
    def __init__(self, config):
        super(SegmentationTrainAgent, self).__init__(config)
        assert(self.config_sampler['patch_mode'] <=3)
            
    def get_output_and_loss(self):
        self.class_num = self.config_net['class_num']
        multi_scale_loss = self.config_train.get('multi_scale_loss', False)
        gradual_train    = self.config_train.get('gradual_train', False)
        size_constraint  = self.config_train.get('size_constraint', False)
#        loss_func = SegmentationLoss(n_class=self.class_num)

        full_weight_shape = [x for x in self.full_out_shape]
        full_weight_shape[-1] = 1
        self.w = tf.placeholder(tf.float32, shape = full_weight_shape)
        self.y = tf.placeholder(tf.int32, shape = self.full_out_shape)
        
        w_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        b_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        net_class = NetFactory.create(self.config_net['net_type'])
        
        net = net_class(num_classes = self.class_num,
                        parameters  = self.net_params,
                        w_regularizer = w_regularizer,
                        b_regularizer = b_regularizer,
                        name = self.config_net['net_name'])
        predicty = net(self.x, is_training = self.config_net['bn_training'], bn_momentum=self.m)
        self.predicty = tf.reshape(predicty, self.full_out_shape[:-1] + [self.class_num])
        print('network output shape ', self.predicty.shape)
        y_soft  = get_soft_label(self.y, self.class_num)
        loss = self.get_loss(self.predicty, y_soft, self.class_num, weight_map = self.w)
        if(multi_scale_loss):
            print('use soft dice loss')
            
            y_pool1 = tf.nn.pool(y_soft, [1, 2, 2], 'AVG', 'VALID', strides = [1, 2, 2])
            predy_pool1 = tf.nn.pool(self.predicty, [1, 2, 2], 'AVG', 'VALID', strides = [1, 2, 2])
            loss1 =  self.get_loss(predy_pool1, y_pool1, self.class_num)

            y_pool2 = tf.nn.pool(y_soft, [1, 4, 4], 'AVG', 'VALID', strides = [1, 4, 4])
            predy_pool2 = tf.nn.pool(self.predicty, [1, 4, 4], 'AVG', 'VALID', strides = [1, 4, 4])
            loss2 =  self.get_loss(predy_pool2, y_pool2, self.class_num)

            if(gradual_train):
                print('use gradual train')
                self.lambda1 = tf.placeholder(tf.float32, shape = [])
                self.lambda2 = tf.placeholder(tf.float32, shape = [])
                self.lambda3 = tf.placeholder(tf.float32, shape = [])
                loss = self.lambda1 * loss + self.lambda2*loss1 + self.lambda3*loss2
            else:
                loss = (loss + loss1 + loss2 )/3.0
        if(size_constraint):
            print('use size constraint loss')
            size_loss = soft_size_loss(self.predicty, y_soft, self.class_num, weight_map = None)
            loss = loss*0.8 + 0.2*size_loss
        self.loss = loss
        
        pred   = tf.cast(tf.argmax(self.predicty, axis = -1), tf.int32)
        y_reshape = tf.reshape(self.y, tf.shape(pred))
        intersect = tf.cast(tf.reduce_sum(pred * y_reshape), tf.float32)
        volume_sum = tf.cast(tf.reduce_sum(pred) + tf.reduce_sum(y_reshape), tf.float32)
        self.dice = 2.0*intersect/(volume_sum + 1.0)
            
    def get_input_output_feed_dict(self, stage, net_idx = 0):
        while(True):
            if(stage == 'train'):
                try:
                    [x_batch, w_batch, y_batch] = self.sess.run(self.next_train_batch)
                    if (x_batch.shape[0] == self.config_sampler.get('batch_size', 5)):
                        break
                    else:
                        self.sess.run(self.train_init_op)
                except tf.errors.OutOfRangeError:
                    self.sess.run(self.train_init_op)
            else:
                try:
                    [x_batch, w_batch, y_batch] = self.sess.run(self.next_valid_batch[net_idx])
                    if (x_batch.shape[0] == self.config_sampler.get('batch_size', 5)):
                        break
                    else:
                        self.sess.run(self.valid_init_op[net_idx])
                except tf.errors.OutOfRangeError:
                    self.sess.run(self.valid_init_op[net_idx])
        feed_dict = {self.x:x_batch, self.w: w_batch, self.y:y_batch}
        return feed_dict

class RegressionTrainAgent(TrainAgent):
    def __init__(self, config):
        super(RegressionTrainAgent, self).__init__(config)
        assert(self.config_data['patch_mode'] == 2)
    
    def get_output_and_loss(self):
        loss_func = RegressionLoss()
        self.y = tf.placeholder(tf.float32, shape = self.full_out_shape)
        
        w_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        b_regularizer = regularizers.l2_regularizer(self.config_train.get('decay', 1e-7))
        net_class = NetFactory.create(self.config_net['net_type'])

        output_dim_num = np.prod(self.config_sampler['label_shape'])
        net = net_class(num_classes = output_dim_num,
                        parameters = self.net_params,
                        w_regularizer = w_regularizer,
                        b_regularizer = b_regularizer,
                        name = self.config_net['net_name'])
        predicty = net(self.x, is_training = self.config_net['bn_training'], bn_momentum=self.m)
        self.predicty = tf.reshape(predicty, self.full_out_shape)
        print('network output shape ', self.predicty.shape)
        self.loss = loss_func(self.predicty, self.y)
    
    def get_input_output_feed_dict(self):
        [x_batch, y_batch] = self.sess.run(self.next_batch)
        feed_dict = {self.x:x_batch, self.y:y_batch}
        return feed_dict

def model_train(config_file):
    config = parse_config(config_file)
    app_type = config['training']['app_type']
    if(app_type==0):
        train_agent = SegmentationTrainAgent(config)
    else:
        train_agent = RegressionTrainAgent(config)
    train_agent.construct_network()
    train_agent.create_optimization_step_and_data_generator()
    train_agent.train()

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train_test/model_train.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    model_train(config_file)
