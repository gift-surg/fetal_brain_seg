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
    dice_score = 2.0*intersect/(ref_vol + seg_vol)
    dice_score = tf.reduce_mean(dice_score)
    return 1.0-dice_score

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

def soft_size_loss2(prediction, soft_ground_truth, num_class, weight_map = None):
    pred = tf.reshape(prediction, [-1, num_class])
    pred = tf.nn.softmax(pred)
    grnd = tf.reshape(soft_ground_truth, [-1, num_class])

    n_pixel   = tf.cast(tf.reduce_prod(tf.shape(pred))/num_class, tf.float32)
    pred_size = tf.div(tf.reduce_sum(pred, 0), n_pixel)
    grnd_size = tf.div(tf.reduce_sum(grnd, 0), n_pixel)
    size_loss = tf.square(pred_size - grnd_size)
    size_loss = size_loss*tf.constant([0.0] + [1.0]*(num_class-1))
    size_loss = tf.reduce_sum(size_loss)
    size_loss = tf.div(size_loss, (num_class - 1.0))
    return size_loss

def soft_size_loss(prediction, soft_ground_truth, num_class, weight_map = None):
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

def get_variable_list(var_names, include = True):
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
        



def get_input_output_feed_dict(sess, next_train_batch, batch_size, train_init_op):
    while(True):
            try:
                [x_batch, w_batch, y_batch] = sess.run(next_train_batch)
                if (x_batch.shape[0] == batch_size):
                    break
                else:
                    sess.run(train_init_op)
            except tf.errors.OutOfRangeError:
                sess.run(train_init_op)
    return [x_batch, w_batch, y_batch]


def model_train(config_file):
    # load config file
    config = parse_config(config_file)
    config_data    = config['dataset']
    config_sampler = config['sampler']
    config_net     = config['network']
    net_params     = config['network_parameter']
    config_train   = config['training']
    
    seed = config_train.get('random_seed', 1)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # construct network and loss
    batch_size  = config_sampler.get('batch_size', 5)
    full_data_shape = [batch_size] + config_sampler['data_shape']
    full_out_shape  = [batch_size] + config_sampler['label_shape']

    x = tf.placeholder(tf.float32, shape = full_data_shape)
    m = tf.placeholder(tf.float32, shape = []) # momentum for batch normalization
    
    full_weight_shape = [item for item in full_out_shape]
    full_weight_shape[-1] = 1
    w = tf.placeholder(tf.float32, shape = full_weight_shape)
    y = tf.placeholder(tf.int32, shape = full_out_shape)
    
    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(config_net['net_type'])
    
    class_num = config_net['class_num']
    net = net_class(num_classes = class_num,
                    parameters  = net_params,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = config_net['net_name'])
    predicty = net(x, is_training = config_net['bn_training'], bn_momentum=m)
    print('network output shape ', predicty.shape)
    
    multi_scale_loss = config_train.get('multi_scale_loss', False)
    size_constraint  = config_train.get('size_constraint', False)
    loss_func = SegmentationLoss(n_class=class_num)
    loss = loss_func(predicty, y, weight_map = w)
    if(multi_scale_loss):
        y_soft  = get_soft_label(y, class_num)
        print('use soft dice loss')
        loss = soft_dice_loss(predicty, y_soft, class_num)
    
    if(size_constraint):
        print('use size constraint loss')
        y_soft  = get_soft_label(y, class_num)
        loss = loss + soft_size_loss(predicty, y_soft, class_num, weight_map = w)


    learn_rate  = config_train.get('learning_rate', 1e-3)
    vars_fixed  = config_train.get('vars_not_update', None)
    vars_update = get_variable_list(vars_fixed, include = False)
    update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch normalization
    with tf.control_dependencies(update_ops):
        opt_step = tf.train.AdamOptimizer(learn_rate).minimize(loss, var_list = vars_update)
    
    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        train_data = ImageDataGenerator(config_data['data_train'], config_sampler)
        # create an reinitializable iterator given the dataset structure
        train_iterator = Iterator.from_structure(train_data.data.output_types,
                                           train_data.data.output_shapes)
        next_train_batch = train_iterator.get_next()
    # Ops for initializing the two different iterators
    train_init_op = train_iterator.make_initializer(train_data.data)
#    valid_data       = []
#    next_valid_batch = []
#    valid_init_op    = []
#    for i in range(len(self.config_data) - 1):
#        with tf.device('/cpu:0'):
#            temp_valid_data = ImageDataGenerator(self.config_data["data_valid{0:}".format(i)], self.config_sampler)
#            temp_valid_iterator = Iterator.from_structure(temp_valid_data.data.output_types,
#                                        temp_valid_data.data.output_shapes)
#            temp_next_valid_batch = temp_valid_iterator.get_next()
#        temp_valid_init_op = temp_valid_iterator.make_initializer(temp_valid_data.data)
#        valid_data.append(temp_valid_data)
#        next_valid_batch.append(temp_next_valid_batch)
#        valid_init_op.append(temp_valid_init_op)
#    self.valid_data = valid_data
#    self.next_valid_batch = next_valid_batch
#    self.valid_init_op = valid_init_op

    # start the session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    save_vars = get_variable_list([config_net['net_name']], include = True)
    saver = tf.train.Saver(save_vars)
    
    max_iter    = config_train['maximal_iter']
    loss_file   = config_train['model_save_prefix'] + "_loss.txt"
    start_iter  = config_train.get('start_iter', 0)
    loss_list   = []
    if( start_iter > 0):
        vars_not_load = config_train.get('vars_not_load', None)
        restore_vars  = get_variable_list(vars_not_load, include = False)
        restore_saver = tf.train.Saver(restore_vars)
        restore_saver.restore(sess, config_train['pretrained_model'])
    
    # make sure the graph is fixed during training
    tf.get_default_graph().finalize()
    sess.run(train_init_op)
#    for valid_init_op in self.valid_init_op:
#        self.sess.run(valid_init_op)

    for iter in range(start_iter, max_iter):
        # Initialize iterator with the training dataset
        temp_momentum = float(iter-start_iter)/float(max_iter-start_iter)
        [x_batch, w_batch, y_batch] = get_input_output_feed_dict(sess, next_train_batch, batch_size, train_init_op)
        feed_dict = {x: x_batch, y: y_batch, w: w_batch}
        feed_dict[m] = temp_momentum
        opt_step.run(session = sess, feed_dict=feed_dict)

        
        if((iter + 1) % config_train['test_interval'] == 0):
            batch_loss_list = []
            for test_step in range(config_train['test_steps']):
#                step_loss_list = []
                [x_batch, w_batch, y_batch] = get_input_output_feed_dict(sess, next_train_batch, batch_size, train_init_op)
                feed_dict = {x: x_batch, y: y_batch, w: w_batch}
                feed_dict[m] = temp_momentum
                loss_v = loss.eval(feed_dict)
#                step_loss_list.append(loss_v)

#                for valid_idx in range(len(self.config_data) - 1):
#                    feed_dict = self.get_input_output_feed_dict('valid', valid_idx)
#                    feed_dict[self.m] = temp_momentum
#                    loss_v = self.loss.eval(feed_dict)
#                    step_loss_list.append(loss_v)
                batch_loss_list.append(loss_v)
            batch_loss = np.asarray(batch_loss_list, np.float32).mean()
        
            print("{0:} Iter {1:}, loss {2:}".format(datetime.now(), iter+1, batch_loss))
            # save loss and snapshot
            loss_list.append(batch_loss)
            np.savetxt(loss_file, np.asarray(loss_list))
            
        if((iter+1)%config_train['snapshot_iter']  == 0):
            saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(iter+1))
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train_test/model_train.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    model_train(config_file)
