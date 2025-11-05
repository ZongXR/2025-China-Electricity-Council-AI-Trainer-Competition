# -*- coding: utf-8 -*-
import os
import random


if __name__ == "__main__":
    result = []
    for file in os.listdir("/home/bml/storage/dataset/train/Annotations"):
        if file.endswith(".xml"):
            result.append(file.split(".")[0])
    val_labels = random.sample(result, int(0.1*len(result)))
    train_labels = [x for x in result if x not in val_labels]
    with open(os.path.join("/home/bml/storage/dataset/annotations", "train.txt"), "w") as f:
        for name in train_labels:
            f.write(name + "\n")
    with open(os.path.join("/home/bml/storage/dataset/annotations", "val.txt"), "w") as f:
        for name in val_labels:
            f.write(name + "\n")
