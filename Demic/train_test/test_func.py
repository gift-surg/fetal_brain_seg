"""Script for testing
Author: Guotai Wang
"""

import os
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.contrib.data import Iterator
from Demic.net.net_factory import NetFactory
from Demic.image_io.image_loader import ImageLoader
from Demic.image_io.file_read_write import *
from Demic.image_io.image_loader import ImageLoader
from Demic.util.parse_config import parse_config
from Demic.util.image_process import *


def volume_probability_prediction_3d_roi(img, data_shape, label_shape,
                                         class_num, batch_size, sess, x, proby):
    '''
        Test one image with sub regions along x, y, z axis
        img        : a 4D numpy array with shape [D, H, W, C]
        data_shape : input 4d tensor shape
        label_shape: output 4d tensor shape
        class_num  : number of output class
        batch_size : batch size for testing
        sess       : tensorflow session that can run a graph
        x          : input tensor of the graph
        proby      : output tensor of the graph
        '''
    [D, H, W, C] = img.shape
    prob = np.zeros([D, H, W, class_num], np.float32)
    sub_image_patches = []
    sub_image_centers = []
    roid_half = int(label_shape[0]/2)
    roih_half = int(label_shape[1]/2)
    roiw_half = int(label_shape[2]/2)
    
    # get image patches
    for centerd in range(roid_half, D + roid_half, label_shape[0]):
        centerd = min(centerd, D - roid_half)
        for centerh in range(roih_half, H + roih_half, label_shape[1]):
            centerh =  min(centerh, H - roih_half)
            for centerw in range(roiw_half, W + roiw_half, label_shape[2]):
                centerw =  min(centerw, W - roiw_half)
                roi_center = [centerd, centerh, centerw, int(C/2)]
                sub_image_centers.append(roi_center)
                sub_image = extract_roi_from_nd_volume(img, roi_center, data_shape, fill = 'random')
                sub_image_patches.append(sub_image)

    # inference with image patches
    total_batch = len(sub_image_patches)
    max_mini_batch = int((total_batch + batch_size -1)/batch_size)
    for mini_batch_idx in range(max_mini_batch):
        batch_end_idx = min((mini_batch_idx+1)*batch_size, total_batch)
        batch_start_idx = batch_end_idx - batch_size
        data_mini_batch = sub_image_patches[batch_start_idx:batch_end_idx]
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        prob_mini_batch = sess.run(proby, feed_dict = {x:data_mini_batch})
        
        for batch_idx in range(batch_start_idx, batch_end_idx):
            roi_center = sub_image_centers[batch_idx]
            roi_center[3] = int(class_num/2)
            prob = set_roi_to_nd_volume(prob, roi_center, prob_mini_batch[batch_idx-batch_start_idx])
    return prob

def convert_label(in_volume, label_convert_source, label_convert_target):
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if(source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume>0] = convert_volume[mask_volume>0]
    return out_volume
