# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from scipy import ndimage
from Demic.util.image_process import *
from Demic.util.parse_config import parse_config
from Demic.image_io.file_read_write import load_nifty_volume_as_array


def binary_dice3d(s,g):
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    if(not(Ds==Dg and Hs==Hg and Ws==Wg)):
        s = resize_ND_volume_to_given_shape(s, g.shape, order = 0)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = 2.0*s0/(s1 + s2 + 0.00001)
    return dice

def dice_of_binary_volumes(s_name, g_name):
    s = load_nifty_volume_as_array(s_name)
    g = load_nifty_volume_as_array(g_name)
    dice = binary_dice3d(s, g)
    return dice

def dice_evaluation(config_file):
    config = parse_config(config_file)['evaluation']
    labels = config['label_list']
    s_folder = config['segmentation_folder']
    g_folder = config['ground_truth_folder']
    s_postfix = config.get('segmentation_postfix',None)
    g_postfix = config.get('ground_truth_postfix',None)
    remove_outlier = config.get('remove_outlier',False)
    s_postfix = '.nii.gz' if (s_postfix is None) else '_' + s_postfix + '.nii.gz'
    g_postfix = '.nii.gz' if (g_postfix is None) else '_' + g_postfix + '.nii.gz'
    patient_names_file = config['patient_file_names']
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content] 
    dice_all_data = []
    for i in range(len(patient_names)):
        s_name = os.path.join(s_folder, patient_names[i] + s_postfix)
        g_name = os.path.join(g_folder, patient_names[i] + g_postfix)
        s_volume = load_nifty_volume_as_array(s_name)
        g_volume = load_nifty_volume_as_array(g_name)
        s_volume_sub = np.zeros_like(s_volume)
        g_volume_sub = np.zeros_like(g_volume)
        for lab in labels:
            s_volume_sub = s_volume_sub + s_volume == lab
            g_volume_sub = g_volume_sub + g_volume == lab
#        if(s_volume_sub.sum() > 0):
#            s_volume_sub = get_largest_component(s_volume_sub)
        if(remove_outlier):
            strt = ndimage.generate_binary_structure(3,2) # iterate structure
            post = ndimage.morphology.binary_closing(s_volume_sub, strt)
            post = get_largest_component(post)
            s_volume_sub = np.asarray(post*s_volume_sub, np.uint8)
        temp_dice = binary_dice3d(s_volume_sub, g_volume_sub)
        dice_all_data.append(temp_dice)
        print(patient_names[i], temp_dice)
    dice_all_data = np.asarray(dice_all_data)
    dice_mean = [dice_all_data.mean(axis = 0)]
    dice_std  = [dice_all_data.std(axis = 0)]
    np.savetxt(s_folder + '/dice_all.txt', dice_all_data)
    np.savetxt(s_folder + '/dice_mean.txt', dice_mean)
    np.savetxt(s_folder + '/dice_std.txt', dice_std)
    print('dice mean ', dice_mean)
    print('dice std  ', dice_std)
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python util/dice_evaluation.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    dice_evaluation(config_file)
