#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 # windows和Mac下不需要执行该命令
python tools/train.py -c configs/yolo11/yolo11_s_600e_coco.yml --amp --eval --vdl_log_dir=vdl_dir/scalar --use_vdl=true