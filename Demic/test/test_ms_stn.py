
import os
import sys
import numpy as np
import nibabel
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.data import Iterator
import matplotlib.pyplot as plt
from util.parse_config import parse_config
from image_io.file_read_write import *
from image_io.data_generator import ImageDataGenerator
from net.pnet_stn import MultiSliceSpatialTransform

file_name = "../temp/img0.nii"
save_name = "../temp/img0stn.nii"
full_data_shape = [1, 3, 96, 96, 1]
img_raw = load_nifty_volume_as_array(file_name)
img = np.reshape(img_raw, full_data_shape)
print(img.shape)

x = tf.placeholder(tf.float32, shape = full_data_shape)
net = MultiSliceSpatialTransform(full_data_shape)
y = net(x, is_training = False)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
output, param = sess.run(y, feed_dict = {x:img})
output = np.reshape(output, img_raw.shape)
print(output.shape)
print(param)
save_array_as_nifty_volume(output, save_name)
