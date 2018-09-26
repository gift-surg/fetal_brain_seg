
import os
import sys
sys.path.append('../../')
sys.path.append('../../NiftyNet')
import numpy as np
import nibabel
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.data import Iterator
import matplotlib.pyplot as plt
from Demic.util.parse_config import parse_config
from Demic.train_test.train_test_agent import TrainTestAgent
from Demic.image_io.file_read_write import save_array_as_nifty_volume

class CustomTrainTestAgent(TrainTestAgent):
    def train(self):
        # 1, construct network and create data generator
        self._load_data()
        
        # 2, start the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        max_epoch   = self.config_train['maximal_epoch']
        loss_file   = self.config_train['model_save_prefix'] + "_loss.txt"
        start_epoch = self.config_train.get('start_epoch', 0)
        
        temp_dir = './temp'
        for epoch in range(start_epoch, max_epoch):
            # 3, Initialize iterators and train for one epoch
            temp_momentum = float(epoch)/float(max_epoch)
            train_loss_list = []
            self.sess.run(self.train_init_op)
            for step in range(self.config_train['batch_number']):
                one_batch = self.sess.run(self.next_train_batch)
                img = one_batch['image'][0,:,:,:,0]
                wht = one_batch['weight'][0,:,:,:,0]
                lab = one_batch['label'][0,:,:,:,0]
                if(img.shape[0] > lab.shape[0]):
                    wht_pad = np.zeros_like(img, np.float32)
                    margin = (img.shape[0] - lab.shape[0])/2
                    wht_pad[margin:margin + lab.shape[0]] = wht
                    lab_pad = np.zeros_like(img, np.uint8)
                    lab_pad[margin:margin + lab.shape[0]] = lab
                    wht = wht_pad
                    lab = lab_pad
                print(one_batch['image'].shape)
                save_array_as_nifty_volume(img, '{0:}/img{1:}_{2:}.nii'.format(temp_dir, epoch, step))
                save_array_as_nifty_volume(wht, '{0:}/wht{1:}_{2:}.nii'.format(temp_dir, epoch, step))
                save_array_as_nifty_volume(lab, '{0:}/lab{1:}_{2:}.nii'.format(temp_dir, epoch, step))

def model_train(config_file):
    config = parse_config(config_file)
    train_agent = CustomTrainTestAgent(config, 'train')
    train_agent.train()

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python test_image_load.py config.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    model_train(config_file)
