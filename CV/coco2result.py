# -*- coding: utf-8 -*-
import os
import json


if __name__ == '__main__':
    with open("./output/bbox.json", "r") as f:
        test_pred = json.load(f)
    with open("dataset/AutoDrive/annotations/test.json", "r") as f:
        test_json = json.load(f)
    categories = test_json["categories"]
    labels = [x["name"] for x in sorted(categories, key=lambda x: x["id"])]
    images = os.listdir("dataset/AutoDrive/test")
    images = [x for x in images if x.endswith(".jpg")]
    result = dict()
    for x in test_pred:
        image_name = images[x["image_id"]]
        category = labels[x["category_id"]]
        xmin = x["bbox"][0]
        ymin = x["bbox"][1]
        xmax = x["bbox"][2] + xmin
        ymax = x["bbox"][3] + ymin
        score = x["score"]
        if image_name in result.keys():
            result[image_name].append([category, xmin, ymin, xmax, ymax, score])
        else:
            result[image_name] = [[category, xmin, ymin, xmax, ymax, score]]
    with open("output/result.json", "w") as f:
        json.dump(result, f)
