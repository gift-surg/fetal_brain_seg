from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import traceback
import os
import math
import numpy as np
import random
import nibabel
from scipy import ndimage
from Demic.image_io.file_read_write import *
from Demic.image_io.tensor_process import *
from Demic.util.image_process import *


class ImageLoader(object):
    """Wrapper for dataset generation given a read function"""
    def __init__(self, config):
        self.config = config
        # data information
        self.data_root  = config['data_root']
        self.patch_data_type   = config['patch_type_x']
        self.patch_label_type  = config['patch_type_y']
        self.patch_data_shape  = config['patch_shape_x']
        self.patch_label_shape = config['patch_shape_y']
        self.label_convert_source = self.config.get('label_convert_source', None)
        self.label_convert_target = self.config.get('label_convert_target', None)
        self.modality_postfix     = config['modality_postfix']
        self.with_ground_truth   = config.get('with_ground_truth', False)
        self.with_weight    = config.get('with_weight', False)
        self.label_postfix  = config.get('label_postfix', None)
        self.weight_postfix = config.get('weight_postfix', None)
        self.file_postfix   = config['file_postfix']
        self.data_names     = config.get('data_names', None)
        self.data_names_val = config.get('data_names_val', None)
        self.batch_size     = config.get('batch_size', 5)
        
        self.replace_background_with_random = config.get('replace_background_with_random', False)
        self.__check_image_patch_shape()

    def __check_image_patch_shape(self):
        assert(len(self.patch_data_shape) == 4 and len(self.patch_label_shape) == 4)
        patch_label_margin = []
        for i in range(3):
            assert(self.patch_data_shape[i] >= self.patch_label_shape[i])
            margin = (self.patch_data_shape[i] - self.patch_label_shape[i])
            assert( margin % 2 == 0)
            patch_label_margin.append(int(margin/2))
        patch_label_margin.append(0)
        self.patch_label_margin = patch_label_margin

    def __get_patient_names(self, mode):
        """
            mode: train, valid, or test
        """
        if(mode == 'train' or mode == 'test'):
            data_names = self.data_names
        elif(mode =='valid'):
            data_names = self.data_names_val
        else:
            raise ValueError("mode should be train, valid, or test, {0:} received".format(mode))
        assert(os.path.isfile(data_names))
        with open(data_names) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content]
        
        full_patient_names = []
        for i in range(len(patient_names)):
            image_names = []
            for mod_idx in range(len(self.modality_postfix)):
                image_name_short = patient_names[i] + '_' + \
                                self.modality_postfix[mod_idx] + '.' + \
                                self.file_postfix
                image_name = search_file_in_folder_list(self.data_root, image_name_short)
                image_names.append(image_name)
            weight_name = None
            if(self.with_weight):
                weight_name_short = patient_names[i] + '_' + \
                                    self.weight_postfix + '.' + \
                                    self.file_postfix
                weight_name = search_file_in_folder_list(self.data_root, weight_name_short)
            label_name = None
            if(self.with_ground_truth):
                label_name_short = patient_names[i] + '_' + \
                                    self.label_postfix + '.' + \
                                    self.file_postfix
                label_name = search_file_in_folder_list(self.data_root, label_name_short)
            one_patient_names = {}
            one_patient_names['image'] = image_names
            one_patient_names['weight'] = weight_name
            one_patient_names['label']  = label_name
            full_patient_names.append(one_patient_names)
        return patient_names, full_patient_names

    def __load_volumes(self, image_dict):
        """A function used for data map from image names to image arrays
        inputs:
            image_dict: a dictionary of file names
            image_dict['image']: a list of files for images
            image_dict['label']: a 1-ary list of file name for label
            image_dict['weight']: (optional), a 1-ary list of file name for weight
        outputs:
            out_dict: a dictionay of tensors
            out_dict['image']: a 4-d tensor of image
            out_dict['label']: a 4-d tensor of label
            out_dict['weight']: a 4-d tensor of weight
        """
        img_paths= image_dict['image']
        lab_path = image_dict['label']
        imgs   = []
        for i in range(len(self.modality_postfix)):
            img = tf.py_func(load_nifty_volume, [img_paths[i]], tf.float32)
            imgs.append(img)
        imgs = tf.stack(imgs, axis = 3)
        lab = tf.py_func(load_nifty_volume, [lab_path[0]], tf.float32)
        lab = tf.cast(lab, tf.int32)
        lab = tf.stack([lab], axis = 3)
        out_dict = {'image':imgs, 'label':lab}
        if(self.with_weight):
            wht_path = image_dict['weight']
            wht = tf.py_func(load_nifty_volume, [wht_path[0]], tf.float32)
            wht = tf.stack([wht], axis = 3)
            out_dict['weight'] = wht
        return out_dict
    
    def __random_sample_patch(self, img, wht, lab):
        """Sample a patch from the image with a random position.
            The output size of img_slice and label_slice may not be the same. 
            image, weight and label are sampled with the same central voxel.
        """
        # if output shape is larger than input shape, padding is needed
        img_shape_out = self.patch_data_shape
        lab_shape_out = self.patch_label_shape
        img = pad_tensor_to_desired_shape(img, img_shape_out)
        lab = pad_tensor_to_desired_shape(lab, img_shape_out[:-1] + [lab_shape_out[-1]])

        lab_margin    = tf.constant(self.patch_label_margin)
        img_shape_in  = tf.shape(img)
        lab_shape_in  = tf.shape(lab)
        img_shape_sub = tf.subtract(img_shape_in, img_shape_out)
        
        r = tf.random_uniform(tf.shape(img_shape_sub), 0, 1.0)
        img_begin = tf.multiply(tf.cast(img_shape_sub, tf.float32), r)
        img_begin = tf.cast(img_begin, tf.int32)
        img_begin = tf.multiply(img_begin, tf.constant([1, 1, 1, 0]))
        
        lab_begin = img_begin + lab_margin
        lab_begin = tf.multiply(lab_begin, tf.constant([1, 1, 1, 0]))
        
        img_slice = tf.slice(img, img_begin, img_shape_out)
        lab_slice = tf.slice(lab, lab_begin, lab_shape_out)

        if(wht is not None):
            wht = pad_tensor_to_desired_shape(wht, img_shape_out[:-1] + [lab_shape_out[-1]])
            wht_slice = tf.slice(wht, lab_begin, lab_shape_out)
        else:
            wht_slice = None
        return [img_slice, wht_slice, lab_slice]
    
    def __sample_patch(self, input_dict):
        """A function used for data map from orignal image data to augmented and sampled patch
        inputs:
            input_dict: a dictionary of input 4d tensors
            input_dict['image']: 4d tensor of input image
            input_dict['label']: 4d tensor of input label
            input_dict['weight']: (optional) 4d tensor of input weight
        outputs:
            output_dict: a dictionary of output 4d tensors
            output_dict['image']: 4d tensor of sampled image patch
            output_dict['label']: 4d tensor of sampled label patch
            output_dict['weight']: (optional) 4d tensor of sampled weight patch
        """
        image  = input_dict['image']
        label  = input_dict['label']
        weight = input_dict['weight'] if (self.with_weight) else None

        # 1, convert label
        if(self.label_convert_source and self.label_convert_target):
            assert(len(self.label_convert_source) == len(self.label_convert_target))
            label_converted = tf.zeros_like(label)
            for i in range(len(self.label_convert_source)):
                l0 = self.label_convert_source[i]
                l1 = self.label_convert_target[i]
                label_temp = tf.equal(label, tf.multiply(l0, tf.ones_like(label)))
                label_temp = tf.multiply(l1, tf.cast(label_temp,tf.int32))
                label_converted = tf.add(label_converted, label_temp)
            label = label_converted
        
        # 2, augment by random 2D rotation
        random_rotate = self.config.get('random_rotate', None)
        if(random_rotate is not None):
            print('image shape',tf.shape(image))
            assert(len(random_rotate) == 2)
            assert(random_rotate[0] < random_rotate[1])
            angle  = tf.random_uniform([], random_rotate[0], random_rotate[1])
            
            image  = rotate(image, angle, "BILINEAR")
            label  = rotate(label, angle, "NEAREST")
            if(weight is not None):
                weight = rotate(weight,angle, "BILINEAR")

        # 3, extract image patch
        # patch_mode == 0: random sample a pacth inside the image region
        # patch_mode == 1: Sampling with bounding box, crop with 3d bounding box,
        #                  and resize to given size within plane
        patch_mode = self.config.get('patch_mode', 0)
        if(patch_mode == 1):
            margin = self.config.get('bounding_box_margin', [0,0,0])
            [min_idx, max_idx] = get_bounding_box_of_4d_tensor(label, margin+[0])
            image  = crop_4d_tensor_with_bounding_box(image, min_idx, max_idx)
            label  = crop_4d_tensor_with_bounding_box(label, min_idx, max_idx)
            
            new_2d_size = tf.constant(self.config['patch_shape_x'][1:3])
            image  = tf.image.resize_images(image, new_2d_size, method = 0) # bilinear
            label  = tf.image.resize_images(label, new_2d_size, method = 1) # nearest
            if(weight is not None):
                weight = crop_4d_tensor_with_bounding_box(weight, min_idx, max_idx)
                weight = tf.image.resize_images(weight,new_2d_size, method = 0) # bilinear

        [image, weight, label] = self.__random_sample_patch(image, weight, label)
        
        # 4, augment by random 2D flip
        if(self.config.get('flip_left_right', False)):
            [image, weight, label] = random_flip_tensors_in_one_dim([image, weight, label], 2)
        if(self.config.get('flip_up_down', False)):
            [image, weight, label] = random_flip_tensors_in_one_dim([image, weight, label], 1)
        if(weight is None):
            weight = tf.ones_like(label, tf.float32)

        # 5, cast tensor to desired type
        data_type  = tf.float32 if self.patch_data_type=='float32' else tf.int32
        label_type = tf.float32 if self.patch_label_type=='float32' else tf.int32
        image  = tf.cast(image, data_type)
        weight = tf.cast(weight,data_type)
        label  = tf.cast(label, label_type)
        output_dict = {'image': image, 'weight': weight, 'label': label}
        return output_dict

    def __get_test_dataset(self, patient_names, full_patient_names):
        for i in range(len(patient_names)):
            imgs   = []
            for mod in range(len(self.modality_postfix)):
                img = load_nifty_volume(full_patient_names[i]['image'][mod])
                imgs.append(img[0])
            imgs = np.asarray(imgs)
            imgs = np.transpose(imgs, axes = [1, 2, 3, 0])
            one_item = {'image': imgs, 'name':patient_names[i]}
            yield one_item

    def get_dataset(self, mode, shuffle = True):
        patient_names, full_patient_names = self.__get_patient_names(mode)
        print('patient number', len(patient_names), mode)
        img_names, wht_names, lab_names   = [], [], []
        for i in range(len(full_patient_names)):
            img_names.append(full_patient_names[i]['image'])
            wht_names.append([full_patient_names[i]['weight']])
            lab_names.append([full_patient_names[i]['label']])
        if(mode == 'train' or mode == 'valid'):
            dataset = {'image':tf.constant(img_names), 'label':tf.constant(lab_names)}
            if(self.with_weight):
                dataset['weight'] = tf.constant(wht_names)
            dataset = tf.data.Dataset.from_tensor_slices(dataset)
            dataset = dataset.map(self.__load_volumes, num_parallel_calls=5)
            dataset = dataset.map(self.__sample_patch, num_parallel_calls=5)
            dataset = dataset.prefetch(self.batch_size*20)
            if(shuffle):
                dataset = dataset.shuffle(self.batch_size*20)
            dataset = dataset.batch(self.batch_size)
        else:
            dataset = self.__get_test_dataset(patient_names, full_patient_names)
        return dataset
