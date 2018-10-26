# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from niftynet.layer.base_layer import TrainableLayer

BN_COLLECTION = tf.GraphKeys.UPDATE_OPS
class BNLayer(TrainableLayer):
    """
    Batch normalisation layer, with trainable mean value 'beta' and
    std 'gamma'.  'beta' is initialised to 0.0 and 'gamma' is initialised
    to 1.0.  This class assumes 'beta' and 'gamma' share the same type_str of
    regulariser.
    """

    def __init__(self,
                 regularizer=None,
                 moving_decay=0.99,
                 eps=1e-5,
                 name='batch_norm'):
        super(BNLayer, self).__init__(name=name)
        self.eps = eps
        self.moving_decay = moving_decay
        self.regularizers = {'beta': regularizer, 'gamma': regularizer}

    def layer_op(self, inputs, is_training, bn_momentum=0.9):
        outputs = tf.layers.batch_normalization(inputs, momentum=bn_momentum,
                                                 epsilon=self.eps,
                                                 beta_regularizer=self.regularizers['beta'],
                                                 gamma_regularizer=self.regularizers['gamma'],
                                                 training = is_training)
        return outputs

