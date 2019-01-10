# edit PYTHONPATH to make sure it includes the path of NiftyNet and Demic
export PYTHONPATH="/home/guotai/GitHub:/home/guotai/GitHub/NiftyNet-v0.2.0:$PYTHONPATH"
python test.py detect cfg_data_detect.txt
