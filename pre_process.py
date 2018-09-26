import os
import nibabel
import numpy as np
from scipy import ndimage
import sys
from skimage.exposure._adapthist import interpolate
sys.path.append('./Demic')
from image_io.file_read_write import *
from util.image_process import *
from util.dice_evaluation import get_largest_component

def get_bounding_box(volume, margin = (3,8,8)):
    [d_idxes, h_idxes, w_idxes] = np.nonzero(volume>0)
    [D, H, W] = volume.shape
    print(D, H, W, margin, len(d_idxes))
    mind = max(d_idxes.min() - margin[0], 0)
    maxd = min(d_idxes.max() + margin[0], D)
    minh = max(h_idxes.min() - margin[1], 0)
    maxh = min(h_idxes.max() + margin[1], H)
    minw = max(w_idxes.min() - margin[2], 0)
    maxw = min(w_idxes.max() + margin[2], W)
    return [mind, maxd, minh, maxh, minw, maxw]

def crop_volume(volume, roi):
    return volume[np.ix_(range(roi[0], roi[1]),
                         range(roi[2], roi[3]),
                         range(roi[4], roi[5]))]  
                      
def get_patient_names(input_folder):
    file_names = os.listdir(input_folder)
    patient_names = []
    num_p = 0
    num_n = 0
    for x in file_names:
        if("Image.nii.gz" in x):
            patient_name = "_".join(x.split("_")[:-1])
            patient_names.append(patient_name)
            print(patient_name)
            if(patient_name[:3] == "17_"):
                num_p = num_p + 1
            else:
                num_n = num_n + 1
    print("patients volume:", num_p)
    print("normal volume:", num_n)

def get_bbox_and_crop():
    img_folder = '/Users/guotaiwang/Documents/data/FetalBrain/FetalBrain0'
    seg_folder = '/Users/guotaiwang/Documents/workspace/tf_project/fetal_brain_seg/result/pnet-s10-ml-esb'
    save_folder = '/Users/guotaiwang/Dropbox/FetalBrain/auto_seg'
    file_names = os.listdir(seg_folder)
    file_names = [name for name in file_names if name[:3]=='17_' and 'Seg' in name]
    bbox = []
    for lab_name in file_names:
        patient_name = '_'.join(lab_name.split('_')[:-1])
        img_name =  patient_name + '_Image.nii.gz'
        img_full_name = os.path.join(img_folder, img_name)
        lab_full_name = os.path.join(seg_folder, lab_name)
        img_obj = nibabel.load(img_full_name)
        img = img_obj.get_data()
        lab = nibabel.load(lab_full_name).get_data()
        margin = [40, 40, 40]
        [idx_min, idx_max] = get_ND_bounding_box(lab, margin)
        bbox.append(idx_min + idx_max)
        img_sub = crop_ND_volume_with_bounding_box(img, idx_min, idx_max)
        lab_sub = crop_ND_volume_with_bounding_box(lab, idx_min, idx_max)
        img_save_name = os.path.join(save_folder, img_name)
        lab_save_name = os.path.join(save_folder, lab_name)
        
        img_sub_obj = nibabel.Nifti1Image(img_sub, img_obj.affine, img_obj.header)
        lab_sub_obj = nibabel.Nifti1Image(lab_sub, img_obj.affine, img_obj.header)
        nibabel.save(img_sub_obj, img_save_name)
        nibabel.save(lab_sub_obj, lab_save_name)
        print(name, img.shape)
    bbox = np.asarray(bbox)
    np.savetxt(save_folder + '/../bbox.txt', bbox)

def resample_and_crop():
    img_folder = '/Users/guotaiwang/Documents/data/FetalBrain/FetalBrain'
    save_folder = '/Users/guotaiwang/Documents/data/FetalBrain/FetalBrain_bb'
    file_names = os.listdir(img_folder)
    file_names = [name for name in file_names if 'Image' in name]
    for img_name in file_names:
        patient_name = '_'.join(img_name.split('_')[:-1])
        img_name = patient_name + '_Image.nii.gz'
        lab_name = patient_name + '_Label.nii.gz'
        img_full_name = os.path.join(img_folder, img_name)
        lab_full_name = os.path.join(img_folder, lab_name)
        
        img_obj = nibabel.load(img_full_name)
        img     = img_obj.get_data()
        lab     = nibabel.load(lab_full_name).get_data()
        spacing = img_obj.header.get_zooms()
        print('img size', img.shape)
        
        scale = [spacing[0], spacing[1], 1.0]
        img_resample = ndimage.interpolation.zoom(img, scale, order = 1)
        lab_resample = ndimage.interpolation.zoom(lab, scale, order = 0)
        margin = [20, 20, 10]
        [idx_min, idx_max] = get_ND_bounding_box(lab_resample, margin)
        img_sub = crop_ND_volume_with_bounding_box(img_resample, idx_min, idx_max)
        lab_sub = crop_ND_volume_with_bounding_box(lab_resample, idx_min, idx_max)

        img_sub_obj = nibabel.Nifti1Image(img_sub, img_obj.affine, img_obj.header)
        lab_sub_obj = nibabel.Nifti1Image(lab_sub, img_obj.affine, img_obj.header)

        img_save_name = os.path.join(save_folder, img_name)
        lab_save_name = os.path.join(save_folder, lab_name)
        nibabel.save(img_sub_obj, img_save_name)
        nibabel.save(lab_sub_obj, lab_save_name)

def fetal_brain_preprocess(input_folder, output_folder, file_names, \
    img_postfix, lab_postfix, wht_postfix, crop = True):
    with open(file_names) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
    patient_names = ['a22_05']
    for patient_name in patient_names:
        print(patient_name)
        lab_name = os.path.join(input_folder, "{0:}_{1:}.nii.gz".format(patient_name, lab_postfix))
        lab_img = nibabel.load(lab_name)
        lab = lab_img.get_data()
        lab_unique = np.unique(lab)
        assert(len(lab_unique) >=2)
        if(len(lab_unique) > 2):
            lab = lab == 2
        lab = np.asarray(lab > 0, np.uint8)
        assert(lab.sum() > 0)
        bb  = get_bounding_box(lab, (8,8,3))
        [W, H, D] = lab.shape
        roi = [0, W, 0, H] + [bb[4], bb[5]]
        for mod_idx in range(len(img_postfix)):
            mod = img_postfix[mod_idx]
            img_name = os.path.join(input_folder, "{0:}_{1:}.nii.gz".format(patient_name, mod))
            img = nibabel.load(img_name).get_data()
            threshold = 40
            strct = ndimage.generate_binary_structure(3, 5)
            weight = img <= threshold
            weight = ndimage.binary_opening(weight, strct)
            weight = np.asarray(1 - weight, np.float32)
            img_norm = itensity_normalize_one_volume(img, weight)
            if(crop):
                img_norm = crop_volume(img_norm, roi)
            img_norm = nibabel.Nifti1Image(img_norm, lab_img.affine, lab_img.header)
            img_name = os.path.join(output_folder, "{0:}_{1:}.nii.gz".format(patient_name, mod))
            nibabel.save(img_norm, img_name)
            if(mod_idx ==0):
                wht_name = os.path.join(output_folder, "{0:}_{1:}.nii.gz".format(patient_name, wht_postfix))
                if(crop):
                    weight = crop_volume(weight, roi)
                weight = nibabel.Nifti1Image(weight, lab_img.affine, lab_img.header)
                nibabel.save(weight, wht_name)
        if(crop):
            lab = crop_volume(lab, roi)
        lab = nibabel.Nifti1Image(lab, lab_img.affine, lab_img.header)
        lab_name = os.path.join(output_folder, "{0:}_{1:}.nii.gz".format(patient_name, lab_postfix))
        nibabel.save(lab, lab_name)

if __name__ =='__main__':
#    get_bbox_and_crop()
#    resample_and_crop()
    input_folder  = '/Users/guotaiwang/Documents/data/FetalBrain/FetalBrain0'
    output_folder = '/Users/guotaiwang/Documents/data/FetalBrain/FetalBrain'
#    get_patient_names(input_folder)
    file_names = "/Users/guotaiwang/Documents/data/FetalBrain/all_names.txt"
    img_postfix = ['Image']
    lab_postfix = 'Label'
    wht_postfix = 'Weight'
    fetal_brain_preprocess(input_folder, output_folder, file_names, \
        img_postfix, lab_postfix, wht_postfix, crop = False)
