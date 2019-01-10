"""Script for fetal brain detection and segmentation
Author: Guotai Wang
Date:   25 September, 2018
Reference: Michael Ebner et al. An Automated Localization, Segmentation and Reconstruction Framework for Fetal Brain MRI. In MICCAI 2018, pp 313-320.
"""

import os
import sys
import math
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from Demic.train_test.model_test import TestAgent
from Demic.image_io.file_read_write import *
from Demic.util.parse_config import parse_config
from Demic.util.image_process import *

def model_test(net_config_file, data_config_file, detection_only = False):
    config = parse_config(net_config_file)
    config_detect = {}
    config_detect['network'] = config['network1']
    config_detect['network_parameter'] = config['network1_parameter']
    config_detect['testing'] = config['detect_testing']
    detect_agent = TestAgent(config_detect)
    detect_agent.construct_network()
    
    print('construct network finished')
    config_segment = {}
    config_segment['network'] = config['network2']
    config_segment['network_parameter'] = config['network2_parameter']
    config_segment['testing'] = config['segment_testing']
    segment_agent = TestAgent(config_segment)
    segment_agent.construct_network()
    
    class_num = 2
    with open(data_config_file) as f:
        file_names = f.readlines()
    file_names = [file_name.strip()  for file_name in file_names if file_name[0] != '#']
    for item in file_names:
        input_name = item.split(' ')[0]
        output_name = item.split(' ')[1]
        print(input_name)
        # stage 1, detect
        img_dict = load_nifty_volume_as_4d_array(input_name)
        img = img_dict['data_array']
        img = itensity_normalize_one_volume(img, img > 10, True)
        outp = detect_agent.test_one_volume(img)
        out = np.asarray(outp > 0.5, np.uint8)

        if(detection_only):
            margin = [3, 8, 8]
            out = get_detection_binary_bounding_box(out, margin, None, mode = 0)
            save_array_as_nifty_volume(out, output_name, input_name)
            continue

        margin = [3, 20, 20]
        strt = ndimage.generate_binary_structure(3,2) # iterate structure
        post = padded_binary_closing(out, strt)
        post = get_largest_component(post)
        bb_min, bb_max = get_ND_bounding_box(post, margin)

        # stage 2, segment
        img_roi = crop_ND_volume_with_bounding_box(img, bb_min + [0], bb_max + [0])
        outp_roi = segment_agent.test_one_volume(img_roi)
        out_roi = np.asarray(outp_roi > 0.5, np.uint8)
        post = padded_binary_closing(out_roi, strt)
        post = get_largest_component(post)
        out_roi = np.asarray(post*out_roi, np.uint8)
        out = np.zeros(img.shape[:-1], np.uint8)
        out = set_ND_volume_roi_with_bounding_box_range(out, bb_min, bb_max, out_roi)

        save_array_as_nifty_volume(out, output_name, input_name)

if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print('Number of arguments should be 3. e.g.')
        print('    python test.py segment cfg_data_segment.txt')
        exit()
    net_config_file  = 'cfg_net.txt'
    task = sys.argv[1]
    assert(task=='segment' or task == 'detect')
    detection_only = True if task == 'detect' else False
    data_config_file = sys.argv[2]
    assert(os.path.isfile(net_config_file) and os.path.isfile(data_config_file))
    model_test(net_config_file, data_config_file, detection_only)
