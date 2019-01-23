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

def model_test(net_config_file, data_config_file):
    config_net  = parse_config(net_config_file)
    config_data = parse_config(data_config_file)
    config_detect = {}
    config_detect['network'] = config_net['network1']
    config_detect['network_parameter'] = config_net['network1_parameter']
    config_detect['testing'] = config_net['detect_testing']
    detect_agent = TestAgent(config_detect)
    detect_agent.construct_network()
    
    print('construct network finished')
    config_segment = {}
    config_segment['network'] = config_net['network2']
    config_segment['network_parameter'] = config_net['network2_parameter']
    config_segment['testing'] = config_net['segment_testing']
    segment_agent = TestAgent(config_segment)
    segment_agent.construct_network()
    
    class_num = 2
    for item in config_data:
        input_name   = config_data[item]['input']
        detect_name  = config_data[item]['detect_output']
        segment_name = config_data[item]['segment_output']
        print(input_name)
        # stage 1, detect
        img_dict = load_nifty_volume_as_4d_array(input_name)
        img = img_dict['data_array']
        img = itensity_normalize_one_volume(img, img > 10, True)
        outp = detect_agent.test_one_volume(img)
        out = np.asarray(outp > 0.5, np.uint8)

        margin = [3, 8, 8]
        detect_out = get_detection_binary_bounding_box(out, margin, None, mode = 0)
        save_array_as_nifty_volume(detect_out, detect_name, input_name)

        # stage 2, segment
        margin = [0, 10, 10]
        bb_min, bb_max = get_ND_bounding_box(detect_out, margin)

        img_roi = crop_ND_volume_with_bounding_box(img, bb_min + [0], bb_max + [0])
        outp_roi = segment_agent.test_one_volume(img_roi)
        out_roi = np.asarray(outp_roi > 0.5, np.uint8)
        strt = ndimage.generate_binary_structure(3,2)
        post = padded_binary_closing(out_roi, strt)
        post = get_largest_component(post)
        out_roi = out_roi * post
        out = np.zeros(img.shape[:-1], np.uint8)
        out = set_ND_volume_roi_with_bounding_box_range(out, bb_min, bb_max, out_roi)
        save_array_as_nifty_volume(out, segment_name, input_name)

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('    python test.py cfg_data.txt')
        exit()
    net_config_file  = 'cfg_net.txt'
    data_config_file = sys.argv[1]
    assert(os.path.isfile(net_config_file) and os.path.isfile(data_config_file))
    model_test(net_config_file, data_config_file)
