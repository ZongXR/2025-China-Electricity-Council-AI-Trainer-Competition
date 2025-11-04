#!/bin/bash
python dataset/voc/create_list.py -d dataset/voc/
python tools/x2coco.py --dataset_type voc --output_dir dataset/voc2coco --voc_anno_dir dataset/voc/VOCdevkit/VOC2007/Annotations --voc_anno_list dataset/voc/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt --voc_label_list dataset/voc/label_list.txt --voc_out_name voc.json