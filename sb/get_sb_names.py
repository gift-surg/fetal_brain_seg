import os
import sys


def mkdir_if_not_exist(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

def get_mask_names():
    reference_dir = "/mnt/NeuroImage_test"
    data_dir = "/mnt/data/spina_bifida"
    output_dir = "/mnt/NeuroImage_2019"
    patient_names = os.listdir(reference_dir)
    print('patient number {0:}'.format(len(patient_names)))
    for patient_name in patient_names:
        patient_init_dir = "{0:}/{1:}/segmentation_init/seg_manual".format(
             reference_dir, patient_name) 
        file_names = os.listdir(patient_init_dir)
        file_names = [file_name for file_name in file_names if file_name[0]!='.']
        print(patient_name) 
        input_name_list = []
        detect_name_list = []
        segment_name_list = []
        for file_name in file_names:
            input_name = "{0:}/{1:}/nifti/{2:}".format(
                data_dir, patient_name, file_name)
            output_dir_patient = "{0:}/{1:}".format(output_dir, patient_name)
            mkdir_if_not_exist(output_dir_patient)
            mkdir_if_not_exist(output_dir_patient + '/segmentation_init')
            mkdir_if_not_exist(output_dir_patient + '/segmentation_init/detect')
            mkdir_if_not_exist(output_dir_patient + '/segmentation_init/seg_auto')
            detect_name = output_dir_patient + '/segmentation_init/detect/' + file_name
            segment_name= output_dir_patient + '/segmentation_init/seg_auto/' + file_name
            input_name_list.append(input_name)
            detect_name_list.append(detect_name)
            segment_name_list.append(segment_name)
        current_dir = '/home/guotai/projects/fetal_brain_seg'
        cfg_name = '{0:}/sb/{1:}.txt'.format(current_dir,patient_name[:2])
        with open(cfg_name, 'w') as f:
            for i in range(len(input_name_list)):
                f.write('[image_{0:}]\n'.format(i + 1))
                f.write('input = {0:}\n'.format(input_name_list[i]))
                f.write('detect_output = {0:}\n'.format(detect_name_list[i]))
                f.write('segment_output = {0:}\n\n'.format(segment_name_list[i]))
        f.close()
            
if __name__ =='__main__':
    get_mask_names()
