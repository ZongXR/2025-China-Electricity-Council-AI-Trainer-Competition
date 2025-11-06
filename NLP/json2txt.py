# -*- coding: utf-8 -*-
import json


if __name__ == "__main__":
    with open("/home/bml/storage/test100_nolabels.json", "r") as f:
        test_json = json.load(f)
    with open("/home/bml/storage/dataset/test.txt", "w") as f:
        for sample in [x["message"] for x in test_json]:
            f.write(sample.replace("\n", "$") + "\n")