#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 # windows和Mac下不需要执行该命令
python tools/train.py -c /home/bml/storage/PaddleYOLO-ppyoloe-example/configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml --amp --eval --vdl_log_dir=vdl_dir/scalar --use_vdl=true