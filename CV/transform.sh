#!/bin/bash
python /home/bml/storage/PaddleYOLO-ppyoloe-example/tools/x2coco.py  --dataset_type voc --voc_anno_dir /home/bml/storage/dataset/train/Annotations --voc_anno_list /home/bml/storage/dataset/train/train.txt --voc_label_list /home/bml/storage/dataset/train/label_list.txt --output_dir /home/bml/storage/dataset/annotations --voc_out_name train.json
python /home/bml/storage/PaddleYOLO-ppyoloe-example/tools/x2coco.py  --dataset_type voc --voc_anno_dir /home/bml/storage/dataset/train/Annotations --voc_anno_list /home/bml/storage/dataset/train/val.txt --voc_label_list /home/bml/storage/dataset/train/label_list.txt --output_dir /home/bml/storage/dataset/annotations --voc_out_name val.json
mv /home/bml/storage/dataset/train/JPEGImages/*.img /home/bml/storage/dataset/train
mv /home/bml/storage/dataset/test/JPEGImages/*.img /home/bml/storage/dataset/test
rm -rf /home/bml/storage/dataset/train/Annotations
rm -rf /home/bml/storage/dataset/train/JPEGImages
rm -rf /home/bml/storage/dataset/test/JPEGImages