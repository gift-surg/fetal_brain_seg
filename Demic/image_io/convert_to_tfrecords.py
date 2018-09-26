# Created on Wed Oct 11 2017
#
# @author: Guotai Wang
# reference: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

import os
import sys
from scipy import ndimage
import numpy as np
import nibabel
import tensorflow as tf

from Demic.util.parse_config import parse_config
from Demic.image_io.file_read_write import load_nifty_volume_as_array

def search_file_in_folder_list(folder_list, file_name):
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if(os.path.isfile(full_file_name)):
            file_exist = True
            break
    if(file_exist == False):
        raise ValueError('file not exist: {0:}'.format(file_name))
    return full_file_name

class DataLoader():
    def __init__(self, config):
        self.config = config
        # data information
        self.data_root = config['data_root']
        self.modality_postfix = config['modality_postfix']
        self.with_ground_truth  = config.get('with_ground_truth', False)
        self.with_weight = config.get('with_weight', False)
        self.label_postfix  = config.get('label_postfix', None)
        self.weight_postfix = config.get('weight_postfix', None)
        self.file_postfix = config['file_post_fix']
        self.data_names = config.get('data_names', None)
        self.data_subset = config.get('data_subset', None)
        self.replace_background_with_random = config.get('replace_background_with_random', False)

    def __get_patient_names(self):
        if(not(self.data_names is None)):
            assert(os.path.isfile(self.data_names))
            with open(self.data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content] 
        else: # load all image in data_root
            sub_dirs = [x[0] for x in os.walk(self.data_root[0])]
            print(sub_dirs)
            patient_names = []
            for sub_dir in sub_dirs:
                names = os.listdir(sub_dir)
                if(sub_dir == self.data_root[0]):
                    sub_patient_names = []
                    for x in names:
                        if(self.file_postfix in x):
                            idx = x.rfind('_')
                            xsplit = x[:idx]
                            sub_patient_names.append(xsplit)
                else:
                    sub_dir_name = sub_dir[len(self.data_root[0])+1:]
                    sub_patient_names = []
                    for x in names:
                        if(self.file_postfix in x):
                            idx = x.rfind('_')
                            xsplit = os.path.join(sub_dir_name,x[:idx])
                            sub_patient_names.append(xsplit)                    
                sub_patient_names = list(set(sub_patient_names))
                sub_patient_names.sort()
                patient_names.extend(sub_patient_names)   
        return patient_names    

    def load_data(self):
        patient_names = self.__get_patient_names()
        if(not(self.data_subset is None)):
            patient_names = patient_names[self.data_subset[0]:self.data_subset[1]]
        self.patient_names = patient_names
        X = []
        W = []
        Y = []
        spacing = []
        for i in range(len(self.patient_names)):
            print(i, self.patient_names[i])
            if(self.with_weight):
                weight_name_short = self.patient_names[i] + '_' + self.weight_postfix + '.' + self.file_postfix
                weight_name = search_file_in_folder_list(self.data_root, weight_name_short)
                weight = load_nifty_volume_as_array(weight_name)
                w_array = np.asarray([weight], np.float32)
                w_array = np.transpose(w_array, [1, 2, 3, 0]) # [D, H, W, C]
                W.append(w_array)      
            if(self.with_ground_truth):
                label_name_short = self.patient_names[i] + '_' + self.label_postfix + '.' + self.file_postfix
                label_name = search_file_in_folder_list(self.data_root, label_name_short)
                label = load_nifty_volume_as_array(label_name)
                y_array = np.asarray([label])
                y_array = np.transpose(y_array, [1, 2, 3, 0]) # [D, H, W, C]
                Y.append(y_array)  
            volume_list = []
            for mod_idx in range(len(self.modality_postfix)):
                volume_name_short = self.patient_names[i] + '_' + self.modality_postfix[mod_idx] + '.' + self.file_postfix
                volume_name = search_file_in_folder_list(self.data_root, volume_name_short)
                volume, space = load_nifty_volume_as_array(volume_name, with_spacing = True)
                if(self.with_weight and self.replace_background_with_random):
                    arr_random = np.random.normal(0, 1, size = volume.shape)
                    volume[weight==0] = arr_random[weight==0]
                volume_list.append(volume)

            volume_array = np.asarray(volume_list)
            volume_array = np.transpose(volume_array, [1, 2, 3, 0]) # [D, H, W, C]
            X.append(volume_array)
            spacing.append(space)
        print('{0:} volumes have been loaded'.format(len(self.patient_names)))
        self.data   = X
        self.weight = W
        self.label  = Y
        self.spacing = spacing
    
    def get_image_number(self):
        return len(self.patient_names)

    def get_image(self, idx, with_ground_truth = True):
        if(with_ground_truth and self.with_ground_truth):
            label = self.label[idx]
        else:
            label = None
        if(self.with_weight):
            weight = self.weight[idx]
        else:
            weight = None
        output = [self.patient_names[idx], self.data[idx], weight, label, self.spacing[idx]]
        return output

    def save_to_tfrecords(self):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        tfrecords_filename = self.config['tfrecords_filename']
        tfrecord_options= tf.python_io.TFRecordOptions(1)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename, tfrecord_options)
        for i in range(len(self.data)):
            feature_dict = {}
            img    = np.asarray(self.data[i], np.float32)
            img_raw    = img.tostring()
            img_shape    = np.asarray(img.shape, np.int32)
            img_shape_raw    = img_shape.tostring()
            feature_dict['image_raw'] = _bytes_feature(img_raw)
            feature_dict['image_shape_raw'] = _bytes_feature(img_shape_raw)
            if(self.with_weight):
                weight = np.asarray(self.weight[i], np.float32)
                weight_raw = weight.tostring()
                weight_shape = np.asarray(weight.shape, np.int32)
                weight_shape_raw = weight_shape.tostring()
                feature_dict['weight_raw'] = _bytes_feature(weight_raw)
                feature_dict['weight_shape_raw'] = _bytes_feature(weight_shape_raw)
            if(self.with_ground_truth):
                label  = np.asarray(self.label[i], np.int32)
                label_raw  = label.tostring()
                label_shape  = np.asarray(label.shape, np.int32)
                label_shape_raw  = label_shape.tostring()
                feature_dict['label_raw'] = _bytes_feature(label_raw)
                feature_dict['label_shape_raw'] = _bytes_feature(label_shape_raw)
            example = tf.train.Example(features=tf.train.Features(feature = feature_dict))
            writer.write(example.SerializeToString())
        writer.close()

def convert_to_rf_records(config_file):
    config = parse_config(config_file)
    config_data = config['data']
    data_loader = DataLoader(config_data)
    data_loader.load_data()
    data_loader.save_to_tfrecords()
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python convert_to_tfrecords.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    convert_to_rf_records(config_file)
