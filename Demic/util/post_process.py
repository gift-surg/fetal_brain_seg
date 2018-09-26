# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from scipy import ndimage
from Demic.util.image_process import get_largest_component
from Demic.util.parse_config import parse_config
from Demic.image_io.file_read_write import *


def post_process(config_file):
    config = parse_config(config_file)['post_process']
    in_folder  = config['input_folder']
    out_folder = config['output_folder']
    s_postfix = config.get('segmentation_postfix',None)
    s_postfix = '.nii.gz' if (s_postfix is None) else '_' + s_postfix + '.nii.gz'

    file_names = os.listdir(in_folder)
    file_names = [name for name in file_names if 'nii.gz' in name]

    for file_name in file_names:
        in_name  = os.path.join(in_folder, file_name)
        out_name = os.path.join(out_folder, file_name)
        seg = load_nifty_volume_as_array(in_name)
        strt = ndimage.generate_binary_structure(3,2) # iterate structure
        post = ndimage.morphology.binary_closing(seg, strt)
        post = get_largest_component(post)
        post = np.asarray(post*seg, np.uint8)
        save_array_as_nifty_volume(post, out_name)

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python util/post_process.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    post_process(config_file)
