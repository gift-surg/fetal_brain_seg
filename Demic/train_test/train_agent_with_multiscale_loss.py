"""Script for training
Author: Guotai Wang
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.data import Iterator
from niftynet.layer.loss_segmentation import LossFunction as SegmentationLoss
from Demic.train_test.train_agent import TrainAgent


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

class TrainAgentWithMultiScaleLoss(TrainAgent):
    def __init__(self, config):
        super(TrainAgentWithMultiScaleLoss, self).__init__(config)
    
    def set_loss_and_optimizer(self):
        # 1, set loss function
        print('use multi-scale loss')
        loss_type = self.config_net.get('loss_type', 'Dice')
        loss_func = SegmentationLoss(self.class_num, loss_type)
        self.loss = loss_func(self.predicty, self.y, weight_map = self.w)

        loss_func1 = SegmentationLoss(self.class_num, loss_type)
        loss_func2 = SegmentationLoss(self.class_num, loss_type)
        loss_func3 = SegmentationLoss(self.class_num, loss_type)
    
        y_soft  = get_soft_label(self.y, self.class_num)
        y_pool1 = tf.nn.pool(y_soft, [1, 3, 3], 'AVG', 'VALID', strides = [1, 3, 3])
        predy_pool1 = tf.nn.pool(self.predicty, [1, 3, 3], 'AVG', 'VALID', strides = [1, 3, 3])
        loss1 = soft_dice_loss(predy_pool1, y_pool1, self.class_num)

        y_pool2 = tf.nn.pool(y_soft, [1, 6, 6], 'AVG', 'VALID', strides = [1, 6, 6])
        predy_pool2 = tf.nn.pool(self.predicty, [1, 6, 6], 'AVG', 'VALID', strides = [1, 6, 6])
        loss2 = soft_dice_loss(predy_pool2, y_pool2, self.class_num)

        y_pool3 = tf.nn.pool(y_soft, [1, 12, 12], 'AVG', 'VALID', strides = [1, 12, 12])
        predy_pool3 = tf.nn.pool(self.predicty, [1, 12, 12], 'AVG', 'VALID', strides = [1, 12, 12])
        loss3 = soft_dice_loss(predy_pool3, y_pool3, self.class_num)
        self.loss = (self.loss + loss1 + loss2 + loss3)/4.0
        
        # 2, set optimizer
        lr = self.config_train.get('learning_rate', 1e-3)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch normalization
        with tf.control_dependencies(update_ops):
            self.opt_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
