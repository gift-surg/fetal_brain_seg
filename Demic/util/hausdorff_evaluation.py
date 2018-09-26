# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import math
import random
import numpy as np
from scipy import ndimage
from Demic.util.image_process import *
from Demic.util.parse_config import parse_config
from Demic.image_io.file_read_write import load_nifty_volume_as_array


def binary_hausdorff3d(s, g, spacing):
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    if(not(Ds==Dg and Hs==Hg and Ws==Wg)):
        s = resize_ND_volume_to_given_shape(s, g.shape, order = 0)
    scale = [1.0, spacing[1], spacing[2]]
    s_resample = ndimage.interpolation.zoom(s, scale, order = 0)
    g_resample = ndimage.interpolation.zoom(g, scale, order = 0)
    point_list_s = volume_to_surface(s_resample)
    point_list_g = volume_to_surface(g_resample)
    new_spacing = [spacing[0], 1.0, 1.0]
    dis1 = hausdorff_distance_from_one_surface_to_another(point_list_s, point_list_g, new_spacing)
    dis2 = hausdorff_distance_from_one_surface_to_another(point_list_g, point_list_s, new_spacing)
    return max(dis1, dis2)

def hausdorff_distance_from_one_surface_to_another(point_list_s, point_list_g, spacing):
    dis_square = 0.0
    n_max = 300
    if(len(point_list_s) > n_max):
        point_list_s = random.sample(point_list_s, n_max)
    for ps in point_list_s:
        ps_nearest = 1e10
        for pg in point_list_g:
            dd = spacing[0]*(ps[0] - pg[0])
            dh = spacing[1]*(ps[1] - pg[1])
            dw = spacing[2]*(ps[2] - pg[2])
            temp_dis_square = dd*dd + dh*dh + dw*dw
            if(temp_dis_square < ps_nearest):
                ps_nearest = temp_dis_square
        if(dis_square < ps_nearest):
            dis_square = ps_nearest
    return math.sqrt(dis_square)

def volume_to_surface(img):
    strt = ndimage.generate_binary_structure(3,2)
    img  = ndimage.morphology.binary_closing(img, strt, 5)
    point_list = []
    [D, H, W] = img.shape
    offset_d  = [-1, 1,  0, 0,  0, 0]
    offset_h  = [ 0, 0, -1, 1,  0, 0]
    offset_w  = [ 0, 0,  0, 0, -1, 1]
    for d in range(1, D-1):
        for h in range(1, H-1):
            for w in range(1, W-1):
                if(img[d, h, w] > 0):
                    edge_flag = False
                    for idx in range(6):
                        if(img[d + offset_d[idx], h + offset_h[idx], w + offset_w[idx]] == 0):
                            edge_flag = True
                            break
                    if(edge_flag):
                        point_list.append([d, h, w])
    return point_list

def hausdorff_evaluation(config_file):
    random.seed(0)
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
    score_all_data = []
    for i in range(len(patient_names)):
        s_name = os.path.join(s_folder, patient_names[i] + s_postfix)
        g_name = os.path.join(g_folder, patient_names[i] + g_postfix)
        i_name = os.path.join(g_folder, patient_names[i] + '_Image.nii.gz')
        s_volume = load_nifty_volume_as_array(s_name)
        g_volume = load_nifty_volume_as_array(g_name)
        i_volume, spacing = load_nifty_volume_as_array(i_name, with_spacing = True)
        s_volume_sub = np.zeros_like(s_volume)
        g_volume_sub = np.zeros_like(g_volume)
        for lab in labels:
            s_volume_sub = s_volume_sub + s_volume == lab
            g_volume_sub = g_volume_sub + g_volume == lab
        if(remove_outlier):
            strt = ndimage.generate_binary_structure(3,2) # iterate structure
            post = ndimage.morphology.binary_closing(s_volume_sub, strt)
            post = get_largest_component(post)
            s_volume_sub = np.asarray(post*s_volume_sub, np.uint8)
        temp_score = binary_hausdorff3d(s_volume_sub, g_volume_sub, spacing)
        score_all_data.append(temp_score)
        print(patient_names[i], temp_score)
    score_all_data = np.asarray(score_all_data)
    score_mean = [score_all_data.mean(axis = 0)]
    score_std  = [score_all_data.std(axis = 0)]
    score_postfix = config['patient_file_names'].split('/')[-1].split('.')[0]
    np.savetxt(s_folder + '/hausdorff_{0:}.txt'.format(score_postfix), score_all_data)
    np.savetxt(s_folder + '/hausdorff_{0:}_mean.txt'.format(score_postfix), score_mean)
    np.savetxt(s_folder + '/hausdorff_{0:}_std.txt'.format(score_postfix), score_std)
    print('hausdorff mean ', score_mean)
    print('hausdorff std  ', score_std)
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python util/hausdorff_evaluation.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    hausdorff_evaluation(config_file)
