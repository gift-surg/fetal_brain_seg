"""Script for testing
Author: Guotai Wang
"""

import os
import sys
from Demic.image_io.file_read_write import *


def convert_image():
    img_names = ['T2_Brain_501', 'T2_Brain_601', 'T2_Brain_701', 'T2_Brain_801', 'T2_Brain_901', 'T2_Brain_1001', 'T2_Brain_1101', 'T2_spine_1701', 'T2_spine_1801', 'T2_spine_1901', 'T2_spine_2001', 'T2_spine_2101', 'T2_spine_2201', 'T2_spine_2301']
    for img_name in img_names:
        input_name = "../data/06_UZL11-Study1/nifti/{0:}.nii.gz".format(img_name)
        output_name = "../data/06_UZL11-Study1/segmentation_init/temp/nifti/{0:}.nii.gz".format(img_name)
        img = load_nifty_volume_as_array(input_name)
        save_array_as_nifty_volume(img, output_name)
        print(input_name)


def convert_segmentation():
    img_names = ['T2_Brain_501', 'T2_Brain_601', 'T2_Brain_701',  'T2_Brain_901', 'T2_Brain_1001', 'T2_Brain_1101', 'T2_spine_1901']
    seg_folders = ['detect', 'seg_auto', 'seg_manual']
    for img_name in img_names:
        img_raw_name = "../data/06_UZL11-Study1/nifti/{0:}.nii.gz".format(img_name)
        for seg_folder in seg_folders:
            seg_load_name = "../data/06_UZL11-Study1/segmentation_init/temp/{0:}/{1:}.nii.gz".format(seg_folder,img_name)
            seg_save_name = "../data/06_UZL11-Study1/segmentation_init/{0:}/{1:}.nii.gz".format(seg_folder,img_name)
            img_obj = nibabel.load(img_raw_name)
            seg = nibabel.load(seg_load_name).get_data()
            seg_obj = nibabel.Nifti1Image(seg, img_obj.affine, img_obj.header)
            nibabel.save(seg_obj, seg_save_name)

convert_segmentation()
