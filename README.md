# Automatic Fetal Brain Localization and Segmentation 
This repository provides source code and pre-trained models for fetal brain localization and segmentation from fetal MRI. The method is detailed in [1]. If you use any resources in this repository, please cite the following papers:

* [1] Michael Ebner*, Guotai Wang*, Wenqi Li, Michael Aertsen, Premal A. Patel, Rosalind Aughwane, Andrew Melbourne, Tom Doel, Anna L. David, Jan Deprest, Sebastien Ourselin, Tom Vercauteren. "An automated localization, segmentation and reconstruction framework for fetal brain MRI." In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 313-320. 2018.

* [2] Eli Gibson*, Wenqi Li*, Carole Sudre, Lucas Fidon, Dzhoshkun I. Shakir, Guotai Wang, Zach Eaton-Rosen, Robert Gray, Tom Doel, Yipeng Hu, Tom Whyntie, Parashkev Nachev, Marc Modat, Dean C. Barratt, Sébastien Ourselin, M. Jorge Cardoso, Tom Vercauteren.
"NiftyNet: a deep-learning platform for medical imaging." Computer Methods and Programs in Biomedicine, 158 (2018): 113-122. https://arxiv.org/pdf/1709.03485.

*    '*' authors contributed equally.

For image reconstruction code, please refer to https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/NiftyMIC

# Requirements
* A CUDA compatable GPU with memoery not less than 6GB is recommended for training. For testing only, a CUDA compatable GPU may not be required.

* Tensorflow. Install tensorflow following instructions from https://www.tensorflow.org/install. Tested version is 1.7.0.

* NiftyNet. Install it by following instructions from http://niftynet.readthedocs.io/en/dev/installation.html or simply typing:
```bash
pip install niftynet
```

# How to use
## 1, Use pre-trained models for fetal brain localization and segmentation

* For fetal brain localization, run

```bash
python test.py detect cfg_data_detect.txt
```


* For fetal brain segmentation, run

```bash
python test.py segment cfg_data_segment.txt
```

* You can edit `cfg_data_detect.txt` and `cfg_data_segment.txt` to customize the input and output image names.

# Acknowledgement
This work is part of the GIFT-Surg project (https://www.gift-surg.ac.uk/). It is supported by Wellcome Trust [WT101957; 203145Z/16/Z], EPSRC [EP/L016478/1; NS/A000027/1; NS/A000050/1], and the NIHR UCLH BRC.
