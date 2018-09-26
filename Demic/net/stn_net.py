# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.utilities.util_common import look_up_operations
from Demic.net.pnet import PNet
from Demic.net.spatial_transformer import transformer

def w_initializer_near_zero():
    def _initializer(shape, dtype, partition_info):
        stddev = np.sqrt(2.0 / np.prod(shape[:-1]))*1e-3
        from tensorflow.python.ops import random_ops
        return random_ops.truncated_normal(
            shape, 0.0, stddev, dtype=tf.float32)
    return _initializer

class ConvPoolBlock(TrainableLayer):
    def __init__(self,
                 n_chns,
                 kernels,
                 padding = 'SAME',
                 w_initializer = None,
                 w_regularizer = None,
                 pooling = True,
                 acti_func = 'prelu',
                 name = 'ConvPoolBlock'):
        super(ConvPoolBlock, self).__init__(name=name)

        self.kernels = kernels
        self.n_chns  = n_chns
        self.padding = padding
        self.pooling = pooling
        self.acti_func = acti_func
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}
    
    def layer_op(self, input_tensor, is_training, bn_momentum = 0.9):
        output_tensor = input_tensor
        for (kernel_size, n_features) in zip(self.kernels, self.n_chns):
            conv_op = ConvolutionalLayer(n_output_chns=n_features,
                                         kernel_size=kernel_size,
                                         padding = self.padding,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=self.acti_func,
                                         name='{}'.format(n_features))
            output_tensor = conv_op(output_tensor, is_training, bn_momentum)
        if(self.pooling):
            pooling_op = DownSampleLayer('MAX',
                                         kernel_size = 2,
                                         stride = 2,
                                         name='down_2x2')
            output_tensor = pooling_op(output_tensor)
        return output_tensor

class TPPN(TrainableLayer):
    """
        Transformation parameter prediction network
    """
    def __init__(self,
                 num_out,
                 w_initializer = None,
                 w_regularizer = None,
                 b_initializer = None,
                 b_regularizer = None,
                 batch_size   = 5,
                 num_features = [64, 64, 64, 128, 128, 128],
                 acti_func = 'prelu',
                 name = 'TPPN'):
        super(TPPN, self).__init__(name = name)

        self.acti_func = acti_func
        self.num_out = num_out
        self.batch_size = batch_size
        self.num_features = num_features
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        print('using {}'.format(name))

    def layer_op(self, images, is_training, bn_momentum = 0.9, layer_id = -1):
        block1 = ConvPoolBlock((self.num_features[0], self.num_features[0]),
                               ((1,3,3),(1,3,3)),
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               acti_func=self.acti_func,
                               name='B1')
        block2 = ConvPoolBlock((self.num_features[1], self.num_features[1]),
                               ((1,3,3),(1,3,3)),
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               acti_func=self.acti_func,
                               name='B2')

        block3 = ConvPoolBlock((self.num_features[2], self.num_features[2]),
                               ((1,3,3), (1,3,3), (1,3,3)),
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               acti_func=self.acti_func,
                               name='B3')

        block4 = ConvPoolBlock((self.num_features[3], self.num_features[3]),
                               ((1,3,3), (1,3,3)),
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               acti_func=self.acti_func,
                               name='B4')

        block5 = ConvPoolBlock((self.num_features[4], self.num_features[4]),
                               ((1,3,3), (1,3,3)),
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               acti_func=self.acti_func,
                               name='B5')
    
        fc1 = FullyConnectedLayer(self.num_features[5],
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  acti_func=self.acti_func,
                                  name='fc1')
        fc2 = FullyConnectedLayer(self.num_out,
                                  w_initializer=w_initializer_near_zero(),
                                  w_regularizer=self.regularizers['w'],
                                  with_bn = False,
                                  acti_func=self.acti_func,
                                  name='fc2')
        output = block1(images, is_training, bn_momentum)
        output = block2(output, is_training, bn_momentum)
        output = block3(output, is_training, bn_momentum)
        output = block4(output, is_training, bn_momentum)
        output = block5(output, is_training, bn_momentum)

        output = tf.reshape(output, [self.batch_size, -1])
        output = fc1(output, is_training, bn_momentum)
        output = fc2(output, is_training, bn_momentum)
        return output

class MultiSliceSpatialTransform(TrainableLayer):
    def __init__(self,
                 input_shape = [5, 3, 96, 96, 1],
                 w_initializer = None,
                 w_regularizer = None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name = 'MultiSliceSpatialTransform'):
        super(MultiSliceSpatialTransform, self).__init__(name = name)
        self.input_shape = input_shape
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        self.acti_func = acti_func

    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        # convert input tensor into list of stacks
        batch_of_stacks = []
        for i in range(self.input_shape[0]):
            temp_stack = []
            for j in range(self.input_shape[1]):
                begin = [i, j, 0, 0, 0]
                size  = [item for item in self.input_shape]
                size[0] = 1
                size[1] = 1
                temp_slice  = tf.slice(images, begin, size)
                temp_stack.append(temp_slice)
            batch_of_stacks.append(temp_stack)
        central_j = int((self.input_shape[1]-1)/2)

        # get input for tpp, with size of [N*D, H, W, 2]
        batch_of_cat_stacks = []
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                temp_concate = tf.concat([batch_of_stacks[i][j], batch_of_stacks[i][central_j]], -1)
                batch_of_cat_stacks.append(temp_concate)
        tpp_input = tf.concat(batch_of_cat_stacks, axis = 0)

        # get stn paramters with TPPN
        tpp_batch = self.input_shape[0] * self.input_shape[1]
        tpp_net = TPPN(3,
                       self.initializers['w'],
                       self.regularizers['w'],
                       self.initializers['b'],
                       self.regularizers['b'],
                       batch_size = tpp_batch,
                       name = 'tpp_net')
        raw_param = tpp_net(tpp_input, is_training, bn_momentum)
        
        # convert stn_param from (tpp_batch, 3) to (tpp_batch, 6)
        theta = tf.slice(raw_param, begin=[0,0], size = [tpp_batch, 1])
        dx    = tf.slice(raw_param, begin=[0,1], size = [tpp_batch, 1])
        dy    = tf.slice(raw_param, begin=[0,2], size = [tpp_batch, 1])
        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)
        minus_sin_theta = tf.subtract(tf.zeros_like(sin_theta),sin_theta)
        stn_param = [cos_theta, minus_sin_theta, dx, sin_theta, cos_theta, dy]
        stn_param = tf.concat(stn_param, axis = 1)

        # apply spatial transformation for the input tensor
        stn_input_shape = self.input_shape[1:]
        stn_input_shape[0] = stn_input_shape[0] * self.input_shape[0]
        stn_input = tf.reshape(images, stn_input_shape)

        stn_output = transformer(stn_input, stn_param, (self.input_shape[2],self.input_shape[3]))
        output = tf.reshape(stn_output, self.input_shape)
        return output

if __name__ == '__main__':
    batch_size = 5
    input_shape = [batch_size,3,96,96,1]
    x = tf.placeholder(tf.float32, shape = input_shape)
    net = PNet_STN(2, input_shape = input_shape)
    y = net(x, is_training=True)
    print(y)
