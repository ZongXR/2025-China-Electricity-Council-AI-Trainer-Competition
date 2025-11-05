# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET


if __name__ == "__main__":
    result = set()
    for file in os.listdir("/home/bml/storage/dataset/train/Annotations"):
        if file.endswith(".xml"):
            tree = ET.parse(os.path.join("/home/bml/storage/dataset/train/Annotations", file))
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text
                result.add(name)
    for name in result:
        with open(os.path.join("/home/bml/storage/dataset/annotations", "label_list.txt"), "w") as f:
            f.write(name + "\n")