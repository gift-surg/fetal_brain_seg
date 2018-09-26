# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from scipy import ndimage
from Demic.util.image_process import *
from Demic.util.parse_config import parse_config
from Demic.image_io.file_read_write import load_nifty_volume_as_array


def binary_iou3d(s,g):
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    if(not(Ds==Dg and Hs==Hg and Ws==Wg)):
        s = resize_ND_volume_to_given_shape(s, g.shape, order = 0)
    intersecion = np.multiply(s, g)
    union = np.asarray(s + g >0, np.float32)
    iou = intersecion.sum()/(union.sum() + 1e-10)
    return iou

def iou_of_binary_volumes(s_name, g_name):
    s = load_nifty_volume_as_array(s_name)
    g = load_nifty_volume_as_array(g_name)
    margin = (3, 8, 8)
    g = get_detection_binary_bounding_box(g, margin)
    return binary_iou3d(s, g)

def iou_evaluation(config_file):
    config = parse_config(config_file)['evaluation']
    labels = config['label_list']
    s_folder = config['segmentation_folder']
    g_folder = config['ground_truth_folder']
    s_postfix = config.get('segmentation_postfix',None)
    g_postfix = config.get('ground_truth_postfix',None)
    s_postfix = '.nii.gz' if (s_postfix is None) else '_' + s_postfix + '.nii.gz'
    g_postfix = '.nii.gz' if (g_postfix is None) else '_' + g_postfix + '.nii.gz'
    patient_names_file = config['patient_file_names']
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content] 
    score_all_data = []
    for i in range(len(patient_names)):
        s_name = os.path.join(s_folder, patient_names[i] + s_postfix)
        g_name = os.path.join(g_folder, patient_names[i] + g_postfix)
        temp_iou = iou_of_binary_volumes(s_name, g_name)
        score_all_data.append(temp_iou)
        print(patient_names[i], temp_iou)
    score_all_data = np.asarray(score_all_data)
    score_mean = [score_all_data.mean(axis = 0)]
    score_std  = [score_all_data.std(axis = 0)]
    np.savetxt(s_folder + '/iou_all.txt', score_all_data)
    np.savetxt(s_folder + '/iou_mean.txt', score_mean)
    np.savetxt(s_folder + '/iou_std.txt', score_std)
    print('iou mean ', score_mean)
    print('iou std  ', score_std)
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python util/iou_evaluation.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    iou_evaluation(config_file)
