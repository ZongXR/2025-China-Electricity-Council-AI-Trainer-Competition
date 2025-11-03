#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/yolo11/yolo11_s_600e_coco.yml -o weights=output/yolo11_s_600e_coco/best_model.pdparams --infer_dir=dataset/AutoDrive/test --draw_threshold=0.5
