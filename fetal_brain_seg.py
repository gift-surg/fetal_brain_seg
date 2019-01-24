"""Script for fetal brain detection and segmentation
Author: Guotai Wang
Date:   25 September, 2018
Reference: Michael Ebner et al. An Automated Localization, Segmentation and Reconstruction Framework for Fetal Brain MRI. In MICCAI 2018, pp 313-320.
"""

import os
import sys
import math
import time
from Demic.util.parse_config import parse_config
from test import model_test

if __name__ == '__main__':
    if(('--input_names' not in sys.argv) or 
       ('--segment_output_names' not in sys.argv)):
        print('Inccorrect command line, please follow the format:')
        print('    python fetal_brain_seg.py ' +
              ' --input_names img1.nii.gz img2.nii.gz ...' +
              ' --segment_output_names img1_seg.nii.gz img2_seg.nii.gz ...' + 
              ' (--detect_output_names img1_det.nii.gz img2_det.nii.gz ...')
        exit()
    idx_0 = sys.argv.index('--input_names')
    idx_1 = sys.argv.index('--segment_output_names')
    input_names = sys.argv[idx_0 + 1 : idx_1]
    img_num     = idx_1 - idx_0 - 1
    seg_names   = sys.argv[idx_1 + 1: idx_1 + 1 + img_num]
    det_names   = None
    if('--detect_output_names' in sys.argv):
        idx_2 = sys.argv.index('--detect_output_names')
        det_names = sys.argv[idx_2 + 1 : idx_2 + 1 + img_num]
    data_config = {}
    for i in range(len(input_names)):
        img_i = 'image_{0:}'.format(i)
        data_config[img_i] = {}
        data_config[img_i]['input'] = input_names[i]
        data_config[img_i]['segment_output'] = seg_names[i]
        if(det_names is not None):
            data_config[img_i]['detect_output'] = det_names[i]
    net_config_file  = 'cfg_net.txt'
    net_config  = parse_config(net_config_file)
    model_test(net_config, data_config)
