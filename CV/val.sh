#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/yolo11/yolo11_s_600e_coco.yml -o weights=output/yolo11_s_600e_coco/best_model.pdparams --classwise