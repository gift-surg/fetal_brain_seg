# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.crop import CropLayer
from niftynet.utilities.util_common import look_up_operations

class UNet2DOrigin(TrainableLayer):
    """
        Reimplementation of U-Net
        Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." MICCAI 2015
        The input tensor shape is [N, D, H, W, C] where D is 1
        """
    
    def __init__(self,
                 num_classes,
                 parameters   =None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='UNet2D'):
        super(UNet2DOrigin, self).__init__(name=name)
        
        if(parameters is None):
            self.n_features = [32, 64, 128, 256, 512]
        else:
            self.n_features = parameters.get('num_features', [32, 64, 128, 256, 512])
        self.acti_func = acti_func
        self.num_classes = num_classes
        
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        
        print('using {}'.format(name))
    
    def layer_op(self, images, is_training, bn_momentum=0.9, layer_id=-1):
        x = tf.reshape(images, [-1, 256, 256, 1])
        conv1 = conv_2d(x, 32, 3, activation='relu', padding='same', regularizer="L2")
        conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2")
        pool1 = max_pool_2d(conv1, 2)

        conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2")
        conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2")
        pool2 = max_pool_2d(conv2, 2)

        conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2")
        conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2")
        pool3 = max_pool_2d(conv3, 2)

        conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2")
        conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2")
        pool4 = max_pool_2d(conv4, 2)

        conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2")
        conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2")

        up6 = upsample_2d(conv5,2)
        up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
        conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
        conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

        up7 = upsample_2d(conv6,2)
        up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
        conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
        conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

        up8 = upsample_2d(conv7,2)
        up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
        conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
        conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

        up9 = upsample_2d(conv8,2)
        up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
        conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
        conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

        pred = conv_2d(conv9, 2, 1,  activation='relu', padding='valid')
        return pred
