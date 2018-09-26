import os
import nibabel
import numpy as np
from scipy import ndimage
import sys
sys.path.append('NiftyNet')
from pre_process import crop_volume
from skimage.exposure._adapthist import interpolate
from Demic.util.image_process import itensity_normalize_one_volume
from Demic.image_io.file_read_write import *
from Demic.util.parse_config import parse_config
from Demic.train_test.model_test import TestAgent, set_roi_to_nd_volume

class RawImageTestAgent(TestAgent):
    def infer_one_raw_image(self, full_name, roi):
        # 1, read data and normalize it
        data, spacing = load_nifty_volume_as_array(full_name, with_spacing = True)
        iten_threshold = 40
        strct = ndimage.generate_binary_structure(3, 5)
        weight = data <= iten_threshold
        weight = ndimage.binary_opening(weight, strct)
        weight = 1 - weight
        data_norm = itensity_normalize_one_volume(data, weight)
        
        # 2, crop data and resample it
        sub_data = crop_volume(data_norm, roi)
        sample_factor = [1, spacing[1], spacing[2]]
        data_resample = ndimage.interpolation.zoom(sub_data, sample_factor, order = 3)
        
        # 3, predict the label for sug image, and resample to original resolution
        data_resample = np.reshape(data_resample, list(data_resample.shape) + [1])
        lab = self.test_one_volume(data_resample)
        sample_factor = [1, 1.0/spacing[1], 1.0/spacing[2]]
        lab = ndimage.interpolation.zoom(lab, sample_factor, order = 0)
    
        output = np.zeros_like(data, np.int8)
        roi_center = [(roi[2*i] + roi[2*i+1])/2 for i in range(3)]
        output = set_roi_to_nd_volume(output, roi_center, lab)
    
        output_name_list = full_name.split('/')
        output_name_list[-2] = 'seg'
        otuput_name_str = '/'.join(output_name_list)
        save_array_as_nifty_volume(output, otuput_name_str)

data_root = 'data/test_data/'
config_file = data_root + 'test_pnet.txt'
config = parse_config(config_file)
test_agent = RawImageTestAgent(config)

with open(data_root + 'data_26.txt') as f:
    content = f.readlines()
    patient_names = [x.strip() for x in content]

for name_roi in patient_names:
    name_roi_split = name_roi.split(' ')
    name = name_roi_split[0]
    full_name = data_root + name
    roi  = [int(x) for x in name_roi_split[1:]]
    roi[0] = roi[0] - 1
    roi[2] = roi[2] - 1
    roi[4] = roi[4] - 1
    if '01/' in full_name:
        continue
    test_agent.infer_one_raw_image(full_name, roi)
