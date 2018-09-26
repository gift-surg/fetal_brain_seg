# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer

from niftynet.utilities.util_common import look_up_operations

class VGG21(TrainableLayer):
    """
        Reimplementation of P-Net
        Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." MICCAI 2015
        The input tensor shape is [N, D, H, W, C] where D is 1
        """
    
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='VGG21'):
        super(VGG21, self).__init__(name=name)
        
        self.n_features = [64, 64, 64, 128, 128, 128, 128]
        self.dilations  = [1, 1, 1, 1, 1, 1]
        self.moving_decay = 0.5
        self.acti_func = acti_func
        self.num_classes = num_classes
        
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        block1 = VGGBlock((self.n_features[0], self.n_features[0]),
                            (self.dilations[0], self.dilations[0]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            moving_decay = self.moving_decay,
                            acti_func=self.acti_func,
                            name='B1')
                            
        block2 = VGGBlock((self.n_features[1], self.n_features[1]),
                            (self.dilations[1], self.dilations[1]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            moving_decay = self.moving_decay,
                            acti_func=self.acti_func,
                            name='B2')

        block3 = VGGBlock((self.n_features[2], self.n_features[2]),
                           (self.dilations[2], self.dilations[2]),
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           moving_decay = self.moving_decay,
                           acti_func=self.acti_func,
                           name='B3')
    
        block4 = VGGBlock((self.n_features[3],self.n_features[3],self.n_features[3],self.n_features[3]),
                           ( self.dilations[3], self.dilations[3], self.dilations[3], self.dilations[3]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            moving_decay = self.moving_decay,
                            acti_func=self.acti_func,
                            name='B4')
                            
        block5 = VGGBlock((self.n_features[4],self.n_features[4],self.n_features[4],self.n_features[4]),
                           ( self.dilations[4], self.dilations[4], self.dilations[4], self.dilations[4]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            moving_decay = self.moving_decay,
                            acti_func=self.acti_func,
                            name='B5')
                            
        block6 = VGGBlock((self.n_features[5],self.n_features[5],self.n_features[5],self.n_features[5]),
                           ( self.dilations[5], self.dilations[5], self.dilations[5], self.dilations[5]),
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           moving_decay = self.moving_decay,
                           acti_func=self.acti_func,
                           name='B6')
        
        fc1 = ConvolutionalLayer(n_output_chns=self.n_features[6],
                                     kernel_size=[1,6,6],
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     moving_decay = self.moving_decay,
                                     acti_func=self.acti_func,
                                     name='fc_1')
                                     
        fc2 = ConvolutionalLayer(n_output_chns=self.n_features[6],
                              kernel_size=[1,1,1],
                              w_initializer=self.initializers['w'],
                              w_regularizer=self.regularizers['w'],
                              moving_decay = self.moving_decay,
                              acti_func=self.acti_func,
                              name='fc_2')

        fc3 = ConvLayer(n_output_chns=self.num_classes,
                                 kernel_size=[1,1,1],
                                 w_initializer=self.initializers['w'],
                                 w_regularizer=self.regularizers['w'],
                                 with_bias = True,
                                 name='fc_3')
    
        out = block1(images, is_training, bn_momentum)
        out = block2(out, is_training, bn_momentum)
        out = block3(out, is_training, bn_momentum)
        out = block4(out, is_training, bn_momentum)
        out = block5(out, is_training, bn_momentum)
        out = block6(out, is_training, bn_momentum)
        out = fc1(out, is_training, bn_momentum)
        out = fc2(out, is_training, bn_momentum)
        out = fc3(out)
        return out


class VGGBlock(TrainableLayer):
    def __init__(self,
                 n_chns,
                 dilations,
                 w_initializer=None,
                 w_regularizer=None,
                 moving_decay=0.9,
                 acti_func='relu',
                 name='VGGNet_block'):
        
        super(VGGBlock, self).__init__(name=name)
        
        self.n_chns = n_chns
        self.dilations = dilations
        self.acti_func = acti_func
        self.moving_decay = moving_decay
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}
    
    def layer_op(self, input_tensor, is_training, bn_momentum=0.9):
        output_tensor = input_tensor
        for (n_chn, dilation) in zip(self.n_chns, self.dilations):
            conv_op = ConvolutionalLayer(n_output_chns=n_chn,
                                         kernel_size=[1,3,3],
                                         dilation =[1, dilation, dilation],
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         moving_decay=self.moving_decay,
                                         acti_func=self.acti_func,
                                         name='{}'.format(dilation))
            pool_op = DownSampleLayer('MAX', kernel_size = 2, stride = 2)
            output_tensor = conv_op(output_tensor, is_training, bn_momentum)
            output_tensor = pool_op(output_tensor)
    
        return output_tensor
