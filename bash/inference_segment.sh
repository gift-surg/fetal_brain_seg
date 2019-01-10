# edit PYTHONPATH to make sure it includes the path of NiftyNet and Demic
export PYTHONPATH="/home/guotai/GitHub:/home/guotai/GitHub/NiftyNet-v0.2.0:$PYTHONPATH"
python test.py segment cfg_data_segment.txt
