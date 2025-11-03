#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml -o weights=output/ppyoloe_plus_crn_x_80e_coco/best_model.pdparams --classwise