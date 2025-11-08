#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 # windows和Mac下不需要执行该命令
# 百度R200需要执行以下代码
#export BKCL_PCIE_RING=1
#export BKCL_RING_BUFFER_SIZE=8388608
#export XPU_PADDLE_L3_SIZE=62914560
#export FLAGS_fuse_parameter_memory_size=32
#export FLAGS_fuse_parameter_groups_size=32
#export XPU_BLACK_LIST=reduce_sum,reduce_max,expand_v2
#export FLAGS_selected_xpus=0
python tools/train.py -c /home/bml/storage/PaddleYOLO-ppyoloe-example/configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml --amp --eval --vdl_log_dir=vdl_dir/scalar --use_vdl=true