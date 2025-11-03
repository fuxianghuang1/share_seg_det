#!/bin/bash

# 要运行的 Python 文件的路径
PYTHON_FILE="/home/jiayi/FM_downstream/Segment/main_512.py"

#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='CBIS-DDSM-split'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '4'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30
#
sed -i "s/hypar\['dataset'\]='.*'/hypar['dataset']='INbreast-split'/" $PYTHON_FILE
sed -i "s/hypar\['gpu_id'\]=\[.*\]/hypar['gpu_id']=[0]/" $PYTHON_FILE
sed -i "s/os.environ\['CUDA_VISIBLE_DEVICES'\] = ".*"/os.environ['CUDA_VISIBLE_DEVICES'] = '4'/" $PYTHON_FILE
python $PYTHON_FILE & 
sleep 30
