import os
import nibabel
import numpy as np


def search_file_in_folder_list(folder_list, file_name):
    """ search a file with a part of name in a list of folders
    input:
        folder_list: a list of folders
        file_name:   a substring of a file
    output:
        the full file name
    """
    file_exist = False
    for folder in folder_list:
        full_file_name = os.path.join(folder, file_name)
        if(os.path.isfile(full_file_name)):
            file_exist = True
            break
    if(file_exist == False):
        raise ValueError('file not exist: {0:}'.format(file_name))
    return full_file_name

def load_nifty_volume_as_array(filename, with_spacing=False):
    """Read a nifty image and return data array
    input shape [W, H, D]
    output shape [D, H, W]
    """
    img = nibabel.load(filename)
    data = img.get_data()
    shape = data.shape
    if(len(shape) == 4):
        assert(shape[3] == 1)
        data = np.reshape(data, shape[:-1])
    data = np.transpose(data, [2,1,0])
    if(with_spacing):
        spacing = img.header.get_zooms()
        spacing = [spacing[2], spacing[1], spacing[0]]
        return data, spacing
    return data


def save_array_as_nifty_volume(data, filename):
    """Write a numpy array as nifty image
        numpy data shape [D, H, W]
        nifty image shape [W, H, D]
        """
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)
