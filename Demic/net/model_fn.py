from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from niftynet.network.segmentation.net_factory import NetFactory
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction as SegmentationLoss

def multi_class_dice(predictions, ground_truth, num_classes):
    """Calculate Dice similarity for each class between labels and predictions.
    Inputs:
        predictions (tf.tensor): predictions
        labels (tf.tensor): ground_truth
        num_classes (int): number of classes to calculate the dice
        
    Outputs:
        list: dice coefficient per class
        """
    dice_list = []
    for i in range(num_classes):
        p = tf.cast(tf.equal(predictions,  tf.ones_like(predictions, tf.int32)*i), tf.float32)
        g = tf.cast(tf.equal(ground_truth, tf.ones_like(predictions, tf.int32)*i), tf.float32)
        intersection = tf.reduce_sum(p*g)
        p_volume = tf.reduce_sum(p)
        g_volume = tf.reduce_sum(g)
        dice = 2*intersection/(p_volume + g_volume + 1e-7)
        dice_list.append(dice)
    return dice_list

def model_fn(features, labels, mode, params=None):
    """Construct a tf.estimator.EstimatorSpec. It defines a network and sets loss
       function, optimiser, evaluation ops and custom tensorboard summary ops. 
       For more details, visit
        https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#model_fn.

    Inputs:
        features (tf.Tensor): Tensor of input features to train from. Required
            rank and dimensions are determined by the subsequent ops
            (i.e. the network).
        labels (tf.Tensor): Tensor of training targets or labels. Required
            rank and dimensions are determined by the network output.
        mode (str): One of the tf.estimator.ModeKeys: TRAIN, EVAL or PREDICT
        params (dict, optional): A dictionary to parameterise the model_fn
            (e.g. learning_rate)

    Outputs:
        tf.estimator.EstimatorSpec: A custom EstimatorSpec 
    """
    # 1. get network otuput
    net_config = params['network']
    decay = net_config.get('decay', 1e-7)
    w_regularizer = regularizers.l2_regularizer(decay)
    b_regularizer = regularizers.l2_regularizer(decay)
    if(type(net_config['net_type']) is str):
        net_class = NetFactory.create(net_config['net_type'])
    else:
        print('customized network is used')
        net_class = net_config['net_type']
    net = net_class(num_classes = net_config['class_num'],
                  parameters = params['network_parameter'],
                  w_regularizer = w_regularizer,
                  b_regularizer = b_regularizer,
                  name = net_config['net_name'])
    
    y_score = net(features['x'],
                is_training = mode == tf.estimator.ModeKeys.TRAIN,
                bn_momentum = 0.9)
    y_prob = tf.nn.softmax(y_score)
    y_pred = tf.cast(tf.argmax(y_score, axis = -1), tf.int32)
    predictions = {}
    predictions['score'] = y_score
    predictions['prob' ] = y_prob
    predictions['pred' ] = y_pred

    # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={'out': tf.estimator.export.PredictOutput(predictions)})

    # 2. set up a loss function
    loss_type = net_config.get('loss_type', 'Dice')
    loss_func = SegmentationLoss(net_config['class_num'], loss_type)
    loss = loss_func(y_score,labels['y'])

    # 3. define a training op and ops for updating moving averages
    # (i.e. for batch normalisation)
    global_step = tf.train.get_global_step()
    optim_type  = net_config.get('optimizer', 'Adam')
    lr          = net_config.get('learning_rate', 1e-3)
    momentum    = net_config.get('momentum', 0.9)
    if(optim_type == 'Adam'):
        optimiser = tf.train.AdamOptimizer(lr)
    else:
        optimiser = tf.train.MomentumOptimizer(lr, momentum)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimiser.minimize(loss, global_step=global_step)

    # 4 create custom metric summaries for tensorboard
    dice_tensor = multi_class_dice(y_pred, labels['y'], net_config['class_num'])
    for i in range(net_config['class_num']):
        tf.summary.scalar('dice_cls{}'.format(i), dice_tensor[i])

    # 5. Return EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=None)
