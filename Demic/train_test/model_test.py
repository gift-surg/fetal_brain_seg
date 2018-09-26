"""Script for testing
Author: Guotai Wang
"""

import os
import sys
import math
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from Demic.net.net_factory import NetFactory
from Demic.image_io.file_read_write import *
from Demic.image_io.convert_to_tfrecords import DataLoader
from Demic.util.parse_config import parse_config
from Demic.util.image_process import *

def extract_roi_from_nd_volume(volume, roi_center, roi_shape, fill = 'random'):
    '''Extract an roi from a nD volume
        volume      : input nD numpy array
        roi_center  : center of roi with position
        output_shape: shape of roi
        '''
    input_shape = volume.shape
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = roi_shape)
    else:
        output = np.zeros(roi_shape)
    r0max = [int(x/2) for x in roi_shape]
    r1max = [roi_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], roi_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - roi_center[i]) for i in range(len(r0max))]
    out_center = r0max
    if(len(roi_center)==3):
        output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                      range(out_center[1] - r0[1], out_center[1] + r1[1]),
                      range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                    range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                    range(roi_center[2] - r0[2], roi_center[2] + r1[2]))]
    elif(len(roi_center)==4):
        output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                      range(out_center[1] - r0[1], out_center[1] + r1[1]),
                      range(out_center[2] - r0[2], out_center[2] + r1[2]),
                      range(out_center[3] - r0[3], out_center[3] + r1[3]))] = \
        volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                      range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                      range(roi_center[2] - r0[2], roi_center[2] + r1[2]),
                      range(roi_center[3] - r0[3], roi_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output

def set_roi_to_nd_volume(volume, roi_center, sub_volume):
    '''Set an roi of an ND volume with a sub volume
        volume: an ND numpy array
        roi_center: center of roi
        sub_volume: the sub volume that will be copied
        '''
    volume_shape = volume.shape
    patch_shape  = sub_volume.shape
    output_volume = volume
    for i in range(len(roi_center)):
        if(roi_center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], roi_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - roi_center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(roi_center) == 3):
        output_volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                             range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                             range(roi_center[2] - r0[2], roi_center[2] + r1[2]))] = \
        sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                          range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                          range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(roi_center) == 4):
        output_volume[np.ix_(range(roi_center[0] - r0[0], roi_center[0] + r1[0]),
                             range(roi_center[1] - r0[1], roi_center[1] + r1[1]),
                             range(roi_center[2] - r0[2], roi_center[2] + r1[2]),
                             range(roi_center[3] - r0[3], roi_center[3] + r1[3]))] = \
        sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                          range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                          range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                          range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")
    return output_volume

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
    roid_half0 = int(label_shape[0]/2); roid_half1 = label_shape[0] - roid_half0
    roih_half0 = int(label_shape[1]/2); roih_half1 = label_shape[1] - roih_half0
    roiw_half0 = int(label_shape[2]/2); roiw_half1 = label_shape[2] - roiw_half0
    
    # get image patches
    for centerd in range(roid_half0, D + roid_half0, label_shape[0]):
        centerd = min(centerd, D - roid_half1)
        for centerh in range(roih_half0, H + roih_half0, label_shape[1]):
            centerh =  min(centerh, H - roih_half1)
            for centerw in range(roiw_half0, W + roiw_half0, label_shape[2]):
                centerw =  min(centerw, W - roiw_half1)
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

def get_augment_prediction(pad_img, data_shape, label_shape,
                           class_num, batch_size, sess, x, proby):
    [D, H, W, C] = pad_img.shape
    flip1 = np.flip(pad_img, axis = 2)
    flip2 = np.flip(pad_img, axis = 1)
    flip3 = np.flip(flip1, axis = 1)
    all_input  = np.concatenate((pad_img, flip1, flip2, flip3))
    all_output = volume_probability_prediction_3d_roi(all_input, data_shape, label_shape,
                                               class_num, batch_size, sess, x, proby)
    outputp  = all_output[0:D]
    outputp1 = np.flip(all_output[D:2*D], axis = 2)
    outputp2 = np.flip(all_output[2*D:3*D], axis = 1)
    outputp3 = np.flip(all_output[3*D:4*D], axis = 1)
    outputp3 = np.flip(outputp3, axis = 2)
    return (outputp + outputp1 + outputp2 + outputp3)/4

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

class TestAgent:
    def __init__(self, config):
        self.config_net = config['network']
        self.config_test= config['testing']
        
        # creat net
        net_class = NetFactory.create(self.config_net['net_type'])
        self.net = net_class(num_classes = self.config_net['class_num'],
                             parameters = config['network_parameter'],
                             w_regularizer = None,
                             b_regularizer = None,
                             name = self.config_net['net_name'])

    def construct_network(self):
        batch_size = self.config_test.get('batch_size', 1)
        data_shape = self.config_net['data_shape']
        label_shape= self.config_net['label_shape']
        class_num  = self.config_net['class_num']
        shape_mode = self.config_test.get('shape_mode', 1)
        margin = [data_shape[i] - label_shape[i] for i in range(len(data_shape))]
        
#        full_data_shape  = [None] + data_shape
        full_data_shape = [None, None, None, None, 1]
        self.x = tf.placeholder(tf.float32, shape = full_data_shape)
        print('network input', self.x)
        predicty = self.net(self.x, is_training = True, bn_momentum = 0.0)
        print('network output shape ', predicty)
        self.proby = tf.nn.softmax(predicty)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        all_vars = tf.global_variables()
        restore_vars= [vars for vars in all_vars if self.config_net['net_name'] in vars.name]
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, self.config_net['model_file'])

    def test_one_volume(self, img, test_augment = False):
        # calculate shape of tensors
#        batch_size = self.config_test.get('batch_size', 1)
        data_shape = self.config_net['data_shape']
        label_shape= self.config_net['label_shape']
        class_num  = self.config_net['class_num']
        shape_mode = self.config_test.get('shape_mode', 1)
        margin = [data_shape[i] - label_shape[i] for i in range(len(data_shape))]

        if(shape_mode == 1):
            [D0, H0, W0, C0] = img.shape
            resized_shape = [D0, data_shape[1], data_shape[2], C0]
            resized_img = resize_ND_volume_to_given_shape(img, resized_shape, order = 3)
        else:
            resized_img = img
        [D, H, W, C] = resized_img.shape
        batch_size = 2*D if test_augment else D
        batch_size = min(batch_size, self.config_test['batch_size'])
        # pad input image to desired size
        size_factor = self.config_test.get('size_factor',[1,1,1])
        Dr = int(math.ceil(float(D)/size_factor[0])*size_factor[0])
        Hr = int(math.ceil(float(H)/size_factor[1])*size_factor[1])
        Wr = int(math.ceil(float(W)/size_factor[2])*size_factor[2])
        pad_img = np.random.normal(size = [Dr, Hr, Wr, C])
        pad_img[np.ix_(range(D), range(H), range(W), range(C))] = resized_img
        
        if(shape_mode==3):
            data_shape[0] = Dr
            label_shape[0]= Dr - margin[0]
        data_shape[1] = Hr
        data_shape[2] = Wr
        label_shape[1]= Hr - margin[1]
        label_shape[2]= Wr - margin[2]
        full_data_shape  = [batch_size] + data_shape
        full_label_shape = [batch_size] + label_shape
        full_weight_shape = [i for i in full_data_shape]
        full_weight_shape[-1] = 1
        
        # inference
        if(test_augment):
            outputp = get_augment_prediction(pad_img, data_shape, label_shape,
                                                class_num, batch_size, self.sess, self.x, self.proby)
        else:
            outputp = volume_probability_prediction_3d_roi(pad_img, data_shape, label_shape,
                                                class_num, batch_size, self.sess, self.x, self.proby)
        outputp = outputp[np.ix_(range(D), range(H), range(W), range(class_num))]
        if(shape_mode == 1):
            outputp = resize_ND_volume_to_given_shape(outputp, list(img.shape[:-1]) + [class_num], order = 1)
        return outputp

def model_test(config_file):
    config = parse_config(config_file)
    config_data = config['data']
    data_loader = DataLoader(config_data)
    data_loader.load_data()
    test_agent = TestAgent(config)
    test_agent.construct_network()
    
    class_num = config['network']['class_num']
    crop_z_axis = config['testing']['crop_z_axis']
    detection_only = config['testing'].get('detection_only', False)
    label_source = config_data.get('label_convert_source', None)
    label_target = config_data.get('label_convert_target', None)
    test_augment = config_data.get('test_augment', False)
    test_augment_trans = config_data.get('test_augment_trans', False)
    if(not(label_source is None) and not(label_source is None)):
        assert(len(label_source) == len(label_target))
    img_num = data_loader.get_image_number()
    test_time = []
    print('image number', img_num)
    for i in range(img_num):
        [name, img_raw, weight_raw, lab_raw, spacing] = data_loader.get_image(i)
        if(crop_z_axis):
            roi_min, roi_max = get_ND_bounding_box(lab_raw, margin = [15,20,20,0])
            zmin = roi_min[0]; zmax = roi_max[0]
            img = img_raw[zmin:zmax]
            lab = lab_raw[zmin:zmax]
        if(config_data.get('resize_input', False)):
            t0 = time.time()
            desire_size = [img.shape[0]] + config['network']['data_shape'][1:]
            img_resize = resize_ND_volume_to_given_shape(img, desire_size, order = 1)
            print('resized image', img_resize.shape)
            out_resize = test_agent.test_one_volume(img_resize, test_augment)
            if(test_augment_trans):
                img_trans = np.transpose(img_resize, axes = [0, 2, 1, 3])
                out1 = test_agent.test_one_volume(img_trans, test_augment)
                out1 = np.transpose(out1, axes = [0, 2, 1, 3])
                out_resize = (out_resize + out1)/2
            out = resize_ND_volume_to_given_shape(out_resize, \
                    list(img.shape[:-1]) + [class_num], order = 1)
            out = np.asarray(np.argmax(out, axis = 3), np.int16)
        elif(config_data.get('crop_with_bounding_box', False)):
            assert(config_data.get('with_ground_truth'))
            roi_min, roi_max = get_ND_bounding_box(lab, margin = [5,20,20,0])
            roi_max[3] = img.shape[3] - 1
            img_roi    = crop_ND_volume_with_bounding_box(img, roi_min, roi_max)
            t0 = time.time()
            desire_size    = [img.shape[0]] + config['network']['data_shape'][1:]
            img_roi_resize = resize_ND_volume_to_given_shape(img_roi, desire_size, order = 1)
            out_roi_resize = test_agent.test_one_volume(img_roi_resize, test_augment)
            if(test_augment_trans):
                img_roi_trans = np.transpose(img_roi_resize, axes = [0, 2, 1, 3])
                out_roi_resize1 = test_agent.test_one_volume(img_roi_trans, test_augment)
                out_roi_resize1 = np.transpose(out_roi_resize1, axes = [0, 2, 1, 3])
                out_roi_resize = (out_roi_resize + out_roi_resize1)/2
            out_roi_resize = np.asarray(np.argmax(out_roi_resize, axis = 3), np.int16)
            out_roi = resize_ND_volume_to_given_shape(out_roi_resize, img_roi.shape[:-1], order = 0)
            out = np.zeros(img.shape[:-1], np.uint8)
            out = set_ND_volume_roi_with_bounding_box_range(out, roi_min[:-1],
                    roi_max[:-1], out_roi)
        else:
            t0 = time.time()
            out = test_agent.test_one_volume(img, test_augment)
            if(test_augment_trans):
                img_trans = np.transpose(img, axes = [0, 2, 1, 3])
                out1 = test_agent.test_one_volume(img_trans, test_augment)
                out1 = np.transpose(out1, axes = [0, 2, 1, 3])
                out = (out + out1)/2
            out = np.asarray(np.argmax(out, axis = 3), np.int16)
        test_time.append(time.time() - t0)
        if(not(label_source is None) and not(label_source is None)):
            out = convert_label(out, label_source, label_target)
        if(crop_z_axis):
            out_raw = np.zeros(img_raw.shape[:-1], np.uint8)
            out_raw[zmin:zmax] = out
            out = out_raw
        if(detection_only):
            print('detection only')
            margin = [3, 8, 8]
            out = get_detection_binary_bounding_box(out, margin, spacing, True)

        save_name = '{0:}_{1:}.nii.gz'.format(name, config_data['output_postfix'])
        save_array_as_nifty_volume(out, config_data['save_root']+'/'+save_name)
    test_time = np.asarray(test_time)
    print('test time', test_time.mean(), test_time.std())
    np.savetxt(config_data['save_root'] + '/test_time.txt', test_time)

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train_test/model_test.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    model_test(config_file)
