fatal brain detection and segmentation. 

requirement: tensorflow

#######################
## for brain detection
#######################
1, set detection_only = True  in cfg_net.txt
2, set input and output file names in cfg_data_detect.txt
3, run python test.py cfg_data_detect.txt

#######################
## for brain segmentation
#######################
1, set detection_only = False  in cfg_net.txt
2, set input and output file names in cfg_data.txt
3, run python test.py cfg_data.txt