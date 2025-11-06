#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c /home/bml/storage/PaddleYOLO-ppyoloe-example/configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml -o weights=output/ppyoloe_plus_crn_x_80e_coco/best_model.pdparams --infer_dir=dataset/AutoDrive/test --draw_threshold=0.5  --save_results=true
python ./coco2result.py