import os
import sys
import numpy as np
import nibabel
import tensorflow as tf

from Demic.util.parse_config import parse_config
from Demic.image_io.convert_to_tfrecords import DataLoader

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
