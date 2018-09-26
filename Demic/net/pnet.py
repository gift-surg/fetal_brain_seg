# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.elementwise import ElementwiseLayer

from niftynet.utilities.util_common import look_up_operations

class PNet(TrainableLayer):
    """
        P-Net
            The input tensor shape is [N, D, H, W, C] 
            network parameters:
            -- num_features: features for P-Net, default [64, 64, 64, 64, 64]
            -- dilations:    dilation of P-Net, default [1, 2, 3, 4, 5]
        """
    
    def __init__(self,
                 num_classes,
                 parameters = None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='PNet'):
        super(PNet, self).__init__(name=name)
        if(parameters is None):
            self.n_features = [64, 64, 64, 64, 64]
            self.dilations  = [1, 2, 3, 4, 5]
        else:
            self.n_features = parameters.get('num_features', [64, 64, 64, 64, 64])
            self.dilations  = parameters.get('num_features', [1, 2, 3, 4, 5])
        self.acti_func = acti_func
        self.num_classes = num_classes
        
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        block1 = PNetBlock((self.n_features[0], self.n_features[0]),
                            (self.dilations[0], self.dilations[0]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B1')
                            
        block2 = PNetBlock((self.n_features[1], self.n_features[1]),
                            (self.dilations[1], self.dilations[1]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B2')
                            
        block3 = PNetBlock((self.n_features[2],self.n_features[2],self.n_features[2]),
                           ( self.dilations[2], self.dilations[2], self.dilations[2]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B3')
                            
        block4 = PNetBlock((self.n_features[3],self.n_features[3],self.n_features[3]),
                           ( self.dilations[3], self.dilations[3], self.dilations[3]),
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='B4')
                            
        block5 = PNetBlock((self.n_features[4],self.n_features[4],self.n_features[4]),
                           ( self.dilations[4], self.dilations[4], self.dilations[4]),
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           acti_func=self.acti_func,
                           name='B5')
        
        conv6_1 = ConvolutionalLayer(n_output_chns=self.n_features[0],
                                     kernel_size=[1,3,3],
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     name='conv6_1')
                                     
        conv6_2 = ConvolutionalLayer(n_output_chns=self.num_classes,
                                  kernel_size=[1,3,3],
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  acti_func=self.acti_func,
                                  name='conv6_2')
                                  
        f1 = block1(images, is_training, bn_momentum)
        f2 = block2(f1, is_training, bn_momentum)
        f3 = block3(f2, is_training, bn_momentum)
        f4 = block4(f3, is_training, bn_momentum)
        f5 = block5(f4, is_training, bn_momentum)
        
        fcat = tf.concat((f1, f2, f3, f4, f5), axis = -1)
        f6 = tf.nn.dropout(fcat, 0.8)
        f6 = conv6_1(f6, is_training, bn_momentum)
        f6 = tf.nn.dropout(f6, 0.8)
        f6 = conv6_2(f6, is_training, bn_momentum)
        
        output = f6
        return output


class PNetBlock(TrainableLayer):
    def __init__(self,
                 n_chns,
                 dilations,
                 w_initializer=None,
                 w_regularizer=None,
                 acti_func='relu',
                 name='PNet_block'):
        
        super(PNetBlock, self).__init__(name=name)
        
        self.n_chns = n_chns
        self.dilations = dilations
        self.acti_func = acti_func
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}
    
    def layer_op(self, input_tensor, is_training, bn_momentum):
        output_tensor = input_tensor
        for (n_chn, dilation) in zip(self.n_chns, self.dilations):
            conv_op = ConvolutionalLayer(n_output_chns=n_chn,
                                         kernel_size=[1,3,3],
                                         dilation =[1, dilation, dilation],
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=self.acti_func,
                                         name='{}'.format(dilation))
            output_tensor = conv_op(output_tensor, is_training, bn_momentum)
    
        return output_tensor
