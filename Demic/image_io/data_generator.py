# Created on Wed Oct 11 2017
#
# @author: Guotai Wang
# reference: https://github.com/kratzert/finetune_alexnet_with_tensorflow
"""Containes a helper class for image input pipelines in tensorflow."""
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import TFRecordDataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


def get_one_of_two_tensors(a,b, r):
    """
    randomly return one of two tensors of a and b
    inputs:
        a: an input tensor
        b: an input tensor with the same shape of a
        r: an random number between 0 and 1
    output:
        output: the returned tensor
    """
    r = tf.less(r, tf.constant(0.5))
    r = tf.cast(r, tf.int32)
    a_b = tf.stack([a,b])
    slice_begin = tf.concat([r, tf.zeros_like(tf.shape(a))], -1)
    slice_size  = tf.concat([tf.constant([1]), tf.shape(a)], -1)
    output = tf.slice(a_b, slice_begin, slice_size)
    output = tf.reshape(output, tf.shape(a))
    return output

def random_flip_tensors_in_one_dim(x, d):
    """
    Random flip a tensor in one dimension
    x: a list of tensors
    d: a integer denoting the axis
    """
    r = tf.random_uniform([1], 0, 1)
    r = tf.less(r, tf.constant(0.5))
    r = tf.cast(r, tf.int32)
    y = []
    for xi in x:
        xi_xiflip = tf.stack([xi, tf.reverse(xi, tf.constant([d]))])
        slice_begin = tf.concat([r, tf.zeros_like(tf.shape(xi))], -1)
        slice_size  = tf.concat([tf.constant([1]), tf.shape(xi)], -1)
        flip = tf.slice(xi_xiflip, slice_begin, slice_size)
        flip = tf.reshape(flip, tf.shape(xi))
        y.append(flip)
    return y

class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """
    def __init__(self, data_files, sampler_config):
        """ Create a new ImageDataGenerator.
        
        Receives a configure dictionary, which specify how to load the data
        """
        self.config = sampler_config
        self.__check_image_patch_shape()
        batch_size = self.config['batch_size']
        self.label_convert_source = self.config.get('label_convert_source', None)
        self.label_convert_target = self.config.get('label_convert_target', None)
        
        data = TFRecordDataset(data_files,"ZLIB")
        data = data.map(self._parse_function, num_threads=5,
                        output_buffer_size=20*batch_size)
        if(self.config.get('data_shuffle', False)):
            data = data.shuffle(buffer_size = 20*batch_size)
        data = data.batch(batch_size)
        self.data = data

    def __check_image_patch_shape(self):
        data_shape   = self.config['data_shape']
        weight_shape = self.config['weight_shape']
        label_shape  = self.config['label_shape']
        assert(len(data_shape) == 4 and len(weight_shape) == 4 and len(label_shape) == 4)
        label_margin = []
        for i in range(3):
            assert(label_shape[i] == weight_shape[i])
            assert(data_shape[i] >= label_shape[i])
            margin = (data_shape[i] - label_shape[i]) % 2
            assert( margin == 0)
            label_margin.append(margin)
        label_margin.append(0)
        self.label_margin = label_margin

    def _parse_function(self, example_proto):
        use_predefined_weight = self.config.get('use_predefined_weight', False)
        keys_to_features = {
            'image_raw':tf.FixedLenFeature((), tf.string),
            'label_raw':tf.FixedLenFeature((), tf.string),
            'image_shape_raw':tf.FixedLenFeature((), tf.string),
            'label_shape_raw':tf.FixedLenFeature((), tf.string)}
        if(use_predefined_weight):
            keys_to_features['weight_raw'] = tf.FixedLenFeature((), tf.string)
            keys_to_features['weight_shape_raw'] = tf.FixedLenFeature((), tf.string)
        # parse the data
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        image_shape  = tf.decode_raw(parsed_features['image_shape_raw'],  tf.int32)
        label_shape  = tf.decode_raw(parsed_features['label_shape_raw'],  tf.int32)
        
        image_shape  = tf.reshape(image_shape,  [4])
        label_shape  = tf.reshape(label_shape,  [4])
        image_raw   = tf.decode_raw(parsed_features['image_raw'],  tf.float32)
        label_raw   = tf.decode_raw(parsed_features['label_raw'],  tf.int32)
        image  = tf.reshape(image_raw, image_shape)
        label  = tf.reshape(label_raw, label_shape)

        if(use_predefined_weight):
            weight_shape = tf.decode_raw(parsed_features['weight_shape_raw'], tf.int32)
            weight_shape = tf.reshape(weight_shape, [4])
            weight_raw  = tf.decode_raw(parsed_features['weight_raw'], tf.float32)
            weight = tf.reshape(weight_raw, weight_shape)
        else:
            weight = tf.ones_like(label, tf.float32)

        # extract image patch
        patch_mode = self.config.get('patch_mode', 0)
        if(patch_mode <= 2):
            [img_patch, wht_patch, lab_patch] = self.__get_training_patch(image, weight, label, patch_mode)
        elif(patch_mode == 3):
            [img_patch1, wht_patch1, lab_patch1] = self.__get_training_patch(image, weight, label, 1)
            [img_patch2, wht_patch2, lab_patch2] = self.__get_training_patch(image, weight, label, 2)
            r = tf.random_uniform([1], 0, 1)
            img_patch = get_one_of_two_tensors(img_patch1, img_patch2, r)
            wht_patch = get_one_of_two_tensors(wht_patch1, wht_patch2, r)
            lab_patch = get_one_of_two_tensors(lab_patch1, lab_patch2, r)
        else:
            raise ValueError('unsupported patch mode {0:}'.format(patch_mode))
        return img_patch, wht_patch, lab_patch
    
    def __pad_tensor_to_desired_shape(self, inpt_tensor, outpt_shape):
        """ Pad a tensor to desired shape
            if the input size is larger than output shape, then reture then input tensor
        """
        inpt_shape = tf.shape(inpt_tensor)
        shape_sub = tf.subtract(inpt_shape, outpt_shape)
        flag = tf.cast(tf.less(shape_sub, tf.zeros_like(shape_sub)), tf.int32)
        flag = tf.scalar_mul(tf.constant(-1), flag)
        pad = tf.multiply(shape_sub, flag)
        pad = tf.add(pad, tf.ones_like(pad))
        pad = tf.scalar_mul(tf.constant(0.5), tf.cast(pad, tf.float32))
        pad = tf.cast(pad, tf.int32)
        pad_lr = tf.stack([pad, pad], axis = 1)
        outpt_tensor = tf.pad(inpt_tensor, pad_lr, mode = "REFLECT")
        return outpt_tensor
    
    def __get_4d_bounding_box(self, label, margin):
        """ Get the 4D bounding boxe generated from nonzero region of the label
            if the nonzero region is null, return the tesor size
            """
        # find bounding box first
        max_idx = tf.subtract(tf.shape(label), tf.constant([1,1,1,1], tf.int32))
        margin = tf.constant(margin)
        mask = tf.not_equal(label, tf.constant(0, dtype=tf.int32))
        indices = tf.cast(tf.where(mask), tf.int32)
        indices_min = tf.reduce_min(indices, reduction_indices=[0]) # infinity if indices is null
        indices_min = tf.subtract(indices_min, margin)
        indices_min = tf.maximum(indices_min, tf.constant([0,0,0,0], tf.int32))
        indices_min = tf.minimum(indices_min, max_idx)
        
        indices_max = tf.reduce_max(indices, reduction_indices=[0]) # minus infinity if indces is null
        indices_max = tf.add(indices_max, margin)
        indices_max = tf.minimum(indices_max, tf.subtract(tf.shape(label), tf.constant([1,1,1,1], tf.int32)))
        indices_max = tf.maximum(indices_max, tf.constant([0,0,0,0]))
        
        # in case the nonzero region is null, switch the indices_min and indices_max
        indices_sum = tf.add(indices_min, indices_max)
        indices_sub = tf.subtract(indices_min, indices_max)
        indices_abs = tf.abs(indices_sub)
        indices_min = tf.div(tf.subtract(indices_sum, indices_abs), tf.constant([2,2,2,2]))
        indices_max = tf.div(tf.add(indices_sum, indices_abs), tf.constant([2,2,2,2]))
        return [indices_min, indices_max]
    
    def __crop_4d_tensor_with_bounding_box(self, input_tensor, idx_min, idx_max):
        size = tf.subtract(idx_max, idx_min)
        size = tf.add(size, tf.constant([1,1,1,1], tf.int32))
        output_tensor  = tf.slice(input_tensor, idx_min, size)
        return output_tensor
    
    def __random_sample_patch(self, img, weight, label):
        """Sample a patch from the image with a random position.
            The output size of img_slice and label_slice may not be the same. 
            image, weight and label are sampled with the same central voxel.
        """
        data_shape_out  = tf.constant(self.config['data_shape'])
        weight_shape_out= tf.constant(self.config['weight_shape'])
        label_shape_out = tf.constant(self.config['label_shape'])
        
        # if output shape is larger than input shape, padding is needed
        img = self.__pad_tensor_to_desired_shape(img, data_shape_out)
        weight = self.__pad_tensor_to_desired_shape(weight, weight_shape_out)
        label  = self.__pad_tensor_to_desired_shape(label, label_shape_out)
        
        data_shape_in   = tf.shape(img)
        weight_shape_in = tf.shape(weight)
        label_shape_in  = tf.shape(label)
        
        label_margin    = tf.constant(self.label_margin)
        data_shape_sub = tf.subtract(data_shape_in, data_shape_out)
        
        r = tf.random_uniform(tf.shape(data_shape_sub), 0, 1.0)
        img_begin = tf.multiply(tf.cast(data_shape_sub, tf.float32), r)
        img_begin = tf.cast(img_begin, tf.int32)
        img_begin = tf.multiply(img_begin, tf.constant([1, 1, 1, 0]))
        
        lab_begin = img_begin + label_margin
        lab_begin = tf.multiply(lab_begin, tf.constant([1, 1, 1, 0]))
        
        img_slice    = tf.slice(img, img_begin, data_shape_out)
        weight_slice = tf.slice(weight, img_begin, weight_shape_out)
        label_slice  = tf.slice(label, lab_begin, label_shape_out)

        return [img_slice, weight_slice, label_slice]
    
    def __get_training_patch(self, image, weight, label, patch_mode):
        ## preprocess
        # augmentation by random rotation
        if(patch_mode == 1):
            random_rotate = self.config.get('random_rotate', None)
            if(not(random_rotate is None)):
                assert(len(random_rotate) == 2)
                assert(random_rotate[0] < random_rotate[1])
                angle  = tf.random_uniform([], random_rotate[0], random_rotate[1])
                image  = tf.contrib.image.rotate(image, angle, "BILINEAR")
                weight = tf.contrib.image.rotate(weight, angle,"BILINEAR")
                label  = tf.contrib.image.rotate(label, angle,"NEAREST")
        
        # augmentation by random flip
        if(self.config.get('flip_left_right', False)):
            [image, weight, label] = random_flip_tensors_in_one_dim([image, weight, label], 2)
        if(self.config.get('flip_up_down', False)):
            [image, weight, label] = random_flip_tensors_in_one_dim([image, weight, label], 1)

        # convert label
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


        if(patch_mode == 0):
            # randomly sample patch with fixed size
            [img_slice, weight_slice, label_slice] = self.__random_sample_patch(
                    image, weight, label)
            return img_slice, weight_slice, label_slice
        elif(patch_mode == 1):
            # Sampling with bounding box, crop with 3d bounding box and resize to given size within plane
            margin = self.config.get('bounding_box_margin', [0,0,0])
            [min_idx, max_idx] = self.__get_4d_bounding_box(label, margin+[0])
            img_slice    = self.__crop_4d_tensor_with_bounding_box(image, min_idx, max_idx)
            weight_slice = self.__crop_4d_tensor_with_bounding_box(weight, min_idx, max_idx)
            label_slice  = self.__crop_4d_tensor_with_bounding_box(label, min_idx, max_idx)
            
            new_2d_size = tf.constant(self.config['data_shape'][1:3])
            img_slice   = tf.image.resize_images(img_slice, new_2d_size, method = 0)   # bilinear
            weight_slice= tf.image.resize_images(weight_slice, new_2d_size, method = 0)# bilinear
            label_slice = tf.image.resize_images(label_slice, new_2d_size, method = 1) # nearest
            [img_slice, weight_slice, label_slice] = self.__random_sample_patch(
                     img_slice, weight_slice, label_slice)
            return img_slice, weight_slice, label_slice
        elif(patch_mode == 2):
            # resize 2d images to given size, and get spatial transformer parameters
            new_2d_size = tf.constant(self.config['data_shape'][1:3])
            img_slice   = tf.image.resize_images(image, new_2d_size, method = 0) # bilinear
            weight_slice= tf.image.resize_images(weight,new_2d_size, method = 0) # bilinear
            label_slice = tf.image.resize_images(label, new_2d_size, method = 1) # nearest
            [img_slice, weight_slice, label_slice] = self.__random_sample_patch(
                    img_slice, weight_slice, label_slice)
#            stp = self.__get_spatial_transform_parameter(label_slice)
            return img_slice, weight_slice, label_slice

    def __get_spatial_transform_parameter(self, label):
        """Compute the parameters for spatial transformer (affine)
            [sh,  0, dh]
            [ 0, sw, dw]
        """
        img_size         = tf.cast(tf.shape(label), tf.float32)
        img_size_minus_1 = tf.subtract(img_size, tf.constant([1, 1, 1, 1], tf.float32))
        img_size         = tf.slice(img_size, tf.constant([1]), tf.constant([2]))
        img_size_minus_1 = tf.slice(img_size_minus_1, tf.constant([1]), tf.constant([2]))

        margin = self.config.get('bounding_box_margin', [0,0,0])
        [indices_min, indices_max] = self.__get_4d_bounding_box(label, margin + [0])
        indices_min = tf.slice(indices_min, tf.constant([1]), tf.constant([2]))
        indices_max = tf.slice(indices_max, tf.constant([1]), tf.constant([2]))
        indices_center = tf.add(indices_min, indices_max)
        indices_center = tf.multiply(tf.cast(indices_center, tf.float32),
                                     tf.constant([0.5, 0.5], tf.float32))
        indices_center = tf.divide(indices_center, img_size_minus_1)
        offset = tf.multiply(indices_center, tf.constant([2.0, 2.0], tf.float32))
        offset = tf.subtract(offset, tf.constant([1.0, 1.0], tf.float32))
        offset = tf.expand_dims(offset, 0)
        offset = tf.transpose(offset)
        
        roi_size = tf.cast(tf.subtract(indices_max, indices_min), tf.float32)
        roi_size = tf.add(roi_size, tf.constant([1.0,1.0], tf.float32))
        scale = tf.divide(roi_size, img_size)

        scale_h = tf.multiply(scale, tf.constant([1.0, 0.0], tf.float32))
        scale_w = tf.multiply(scale, tf.constant([0.0, 1.0], tf.float32))
        rotate = tf.stack([scale_h, scale_w])

        param = tf.concat([rotate, offset], 1)
        return param
        
    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot
