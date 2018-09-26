
import os
import sys
sys.path.append('./')
import numpy as np
import nibabel
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.data import Iterator
import matplotlib.pyplot as plt
from Demic.util.parse_config import parse_config
from Demic.image_io.data_generator import ImageDataGenerator

def save_array_as_nifty_volume(data, filename):
    # numpy data shape [D, H, W]
    # nifty image shape [W, H, W]
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)

def test_generator(config_file):
    config = parse_config(config_file)
    config_data = config['tfrecords']
    batch_size  = config_data['batch_size']
    temp_dir    = config_data['temp_dir']

    # Place data loading and preprocessing on the cpu
    tf.set_random_seed(0)
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(config_data)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    #validation_init_op = iterator.make_initializer(val_data.data)

    train_batches_per_epoch = config['training']['batch_number']
    num_epochs = config['training']['maximal_epoch']
    app_type   = config['training']['app_type']
    # Start Tensorflow session
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        total_step = 0
        # Loop over number of epochs
        for epoch in range(num_epochs):
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))
            # Initialize iterator with the training dataset
            sess.run(training_init_op)
            for step in range(train_batches_per_epoch):
                if(app_type==0):
                    [img_batch, weight_batch, label_batch] = sess.run(next_batch)
                    img_0 = img_batch[0,:,:,:, 0]
                    lab_0 = label_batch[0,:,:,:,0]
                    print(epoch, step, img_0.shape, lab_0.shape)
                    img_d = img_0.shape[0]
                    lab_d = lab_0.shape[0]
                    if(lab_d < img_d):
                        margin = (img_d - lab_d)/2
                        pad_lab = np.zeros_like(img_0)
                        pad_lab[np.ix_(range(margin, margin + lab_d), range(lab_0.shape[1]), range(lab_0.shape[2]))] = lab_0
                        lab_0 = pad_lab
                    save_array_as_nifty_volume(img_0, '{0:}/img{1:}.nii'.format(temp_dir, total_step))
                    save_array_as_nifty_volume(lab_0, '{0:}/lab{1:}.nii'.format(temp_dir, total_step))
                else:
                    [img_batch, stp] = sess.run(next_batch)
                    print(stp)
                    img_0 = img_batch[0, 0,:,:, 0]
                    plt.imshow(img_0)
                    plt.show()
                total_step = total_step + 1

def pad_tensor_to_desired_shape(inpt_tensor, outpt_shape):
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

def test_tensor_pad():
    x = tf.placeholder(tf.float32, shape = [12, 20, 20])
    out_shape = tf.constant([20, 20, 20])
    y = pad_tensor_to_desired_shape(x, out_shape)
    with tf.Session() as sess:
        x_v = np.ones(shape = [12, 20, 20])
        feed_dict = {x: x_v}
        y_v = y.eval(feed_dict)
        print(y_v.shape)
        print(y_v)
if __name__ == "__main__":
#    if(len(sys.argv) != 2):
#        print('Number of arguments should be 2. e.g.')
#        print('    python test_generator.py config.txt')
#        exit()
#    config_file = str(sys.argv[1])
#    assert(os.path.isfile(config_file))
#    test_generator(config_file)
    test_tensor_pad()
