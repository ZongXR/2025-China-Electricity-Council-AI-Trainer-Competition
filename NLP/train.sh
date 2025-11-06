#!/bin/bash
python /home/bml/storage/hierachies_example/hierarchical/train.py \
    --dataset_dir "dataset" \
    --save_dir "checkpoint" \
    --device "xpu" \
    --max_seq_length 512 \
    --model_name /home/bml/storage/hierachies_example/hierarchical/ernie-3.0-medium-zh \
    --init_from_ckpt /home/bml/storage/hierachies_example/hierarchical/ernie-3.0-medium-zh/model_state.pdparams \
    --batch_size 32 \
    --early_stop \
    --early_stop_nums 100 \
    --epochs 100