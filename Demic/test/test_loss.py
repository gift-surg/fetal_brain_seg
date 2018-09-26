
import os
import sys
sys.path.append('./')
import numpy as np
import nibabel
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.data import Iterator
import matplotlib.pyplot as plt
from Demic.train_test.model_train import soft_cross_entropy_loss

def test_loss():
    xv = [[10.0, 2.0,
           2.0, 10.0]]
    xv = np.asarray(xv)
    yv = [[1.0, 0.0,
           0.0, 1.0]]
    yv = np.asarray(yv)
    
    x = tf.constant(xv)
    y = tf.constant(yv)
    
    ce= soft_cross_entropy_loss(x, y, 2)
    with tf.Session() as sess:
        cev = ce.eval()
        print(cev)
if __name__ == "__main__":
    test_loss()
