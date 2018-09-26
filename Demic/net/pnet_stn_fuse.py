# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.utilities.util_common import look_up_operations
from Demic.net.pnet import PNet
from Demic.net.stn_net import MultiSliceSpatialTransform

def fuse_layer_w_initializer():
    def _initializer(shape, dtype, partition_info):
        assert(shape[0]==3)
        w_init0 = np.random.rand(shape[1], shape[2], shape[3], shape[4])*1e-5
        w_init2 = np.random.rand(shape[1], shape[2], shape[3], shape[4])*1e-5
        w_init1 = 1 - w_init0 - w_init2
        w_init = np.asarray([w_init0, w_init1, w_init2])
        w_init = tf.constant(w_init, tf.float32)
        return w_init
    return _initializer

def w_initializer_near_zero():
    def _initializer(shape, dtype, partition_info):
        stddev = np.sqrt(2.0 / np.prod(shape[:-1]))*1e-3
        from tensorflow.python.ops import random_ops
        return random_ops.truncated_normal(
            shape, 0.0, stddev, dtype=tf.float32)
    return _initializer

class Fuse_Net(TrainableLayer):
    """
    Fuse_Net, 
    input size: [N, D, H, W, C]
    output size:[N, D, H, W, Cl], Cl is the number of classes
    """
    def __init__(self,
                 num_classes,
                 parameters   =None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='Fuse_Net'):
        super(Fuse_Net, self).__init__(name=name)
        self.parameters   = parameters
        self.acti_func    = acti_func
        self.num_classes  = num_classes
        self.n_features   = [64, 64, 64, 64]
        self.dilations    = [1,  2,  3,  1]
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))

    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        conv1 = ConvolutionalLayer(n_output_chns = self.n_features[0],
                                 kernel_size = [1, 3, 3],
                                 dilation = [1, self.dilations[0], self.dilations[0]],
                                 w_initializer=self.initializers['w'],
                                 w_regularizer=self.regularizers['w'],
                                 name='conv1')
        conv2 = ConvolutionalLayer(n_output_chns = self.n_features[1],
                                 kernel_size = [1, 3, 3],
                                 dilation = [1, self.dilations[1], self.dilations[1]],
                                 w_initializer=self.initializers['w'],
                                 w_regularizer=self.regularizers['w'],
                                 name='conv2')
        conv3 = ConvolutionalLayer(n_output_chns = self.n_features[2],
                                 kernel_size = [1, 3, 3],
                                 dilation = [1, self.dilations[2], self.dilations[2]],
                                 w_initializer=self.initializers['w'],
                                 w_regularizer=self.regularizers['w'],
                                 name='conv3')
        conv4 = ConvolutionalLayer(n_output_chns = self.n_features[3],
                                 kernel_size = [1, 3, 3],
                                 dilation = [1, self.dilations[3], self.dilations[3]],
                                 w_initializer=self.initializers['w'],
                                 w_regularizer=self.regularizers['w'],
                                 name='conv4')
        conv5 = ConvolutionalLayer(n_output_chns = self.num_classes,
                                 kernel_size = [1, 3, 3],
                                 w_initializer=self.initializers['w'],
                                 w_regularizer=self.regularizers['w'],
                                 name='conv5')

        output = conv1(images, is_training, bn_momentum)
        output = conv2(output, is_training, bn_momentum)
        output = conv3(output, is_training, bn_momentum)
        output = conv4(output, is_training, bn_momentum)
        output = conv5(output, is_training, bn_momentum)
        return output


class PNet_STN_DF(TrainableLayer):
    """
        PNet_STN_DF
        The input tensor shape is [N, D, H, W, C]
        network parameters:
        -- input_shape：input shape of network, e.g. [5, 3, 96, 96, 1]
        -- num_features: features for P-Net, default [64, 64, 64, 64, 64]
        -- dilations:    dilation of P-Net, default [1, 2, 3, 4, 5]

        """
    
    def __init__(self,
                 num_classes,
                 parameters   =None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='PNet_STN_DF'):
        super(PNet_STN_DF, self).__init__(name=name)
        self.parameters = parameters
        self.acti_func = acti_func
        self.num_classes = num_classes
        self.input_shape = parameters['input_shape']
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        stn_layer = MultiSliceSpatialTransform(self.input_shape,
                                               w_initializer = self.initializers['w'],
                                               w_regularizer = self.regularizers['w'],
                                               name = 'stn_layer')
        pnet_layer = PNet(self.num_classes,
                          self.parameters,
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name = 'pnet_layer')
        fuse_layer = Fuse_Net(self.num_classes,
                          w_initializer=self.initializers['w'],
                          w_regularizer=self.regularizers['w'],
                          acti_func=self.acti_func,
                          name = 'fuse_layer')

        if (self.parameters['slice_fusion'] == True):
            if(self.parameters['use_stn'] == True):
                aligned = stn_layer(images, is_training, bn_momentum)
            else:
                aligned = images
            output0 = pnet_layer(aligned, is_training, bn_momentum)
            
            [N, D, H, W, C] = self.input_shape
            aligned_reshape = tf.transpose(aligned, perm = [0, 2, 3, 1, 4])
            aligned_reshape = tf.reshape(aligned_reshape, [N, 1, H, W, -1])
            
            output0_reshape = tf.transpose(output0, perm = [0, 2, 3, 1, 4])
            output0_reshape = tf.reshape(output0_reshape, [N, 1, H, W, -1])
            
            fuse_input = tf.concat([aligned_reshape, output0_reshape], axis = -1)
            output = fuse_layer(fuse_input, is_training, bn_momentum)
        else:
            print('slice fusion is false')
            output = pnet_layer(images, is_training, bn_momentum)
        return output

if __name__ == '__main__':
    batch_size = 5
    input_shape = [batch_size,3,96,96,1]
    parameters = {}
    parameters['input_shape'] = input_shape
    x = tf.placeholder(tf.float32, shape = input_shape)
    net = PNet_STN_WDF(2, parameters)
    y = net(x, is_training=True)
    print(y)
