import os
from pathlib import Path
from typing import List
import cv2
import numpy as np
from dataclasses import dataclass

brand_to_index = {
    "adidas0": 0,
    "chanel": 1,
    "gucci": 2,
    "hh": 3,
    "lacoste": 4,
    "mk": 5,
    "nike": 6,
    "prada": 7,
    "puma": 8,
    "supreme": 9
}

index_to_brand = {}
for k, v in brand_to_index.items():
    index_to_brand[v] = k

@dataclass
class LogoBBox:
    logo_id: int
    x: int
    y: int
    w: int
    h: int

def _load_bbox(path):
    bboxes = []
    with open(path, "r") as bbox_file:
         x, y, w, h = [int(num) for num in next(bbox_file).split()]
         bboxes.append([x, y, w, h])
    return bboxes


def load(dataset_root_path, dataset_filename):
    bbox_dir_path = os.path.join(dataset_root_path, "masks")
    image_root_dir_path = os.path.join(dataset_root_path, "jpg")

    video_set = set()
    with open(os.path.join(dataset_root_path, "ImageSets", dataset_filename), 'r') as set_file:
        for line in set_file:
            video_set.add(line.strip())

    images = []
    logos = []
    for dir_name in os.listdir(image_root_dir_path):
        dir_path = os.path.join(image_root_dir_path, dir_name)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.splitext(file_name)[0] in video_set:
                img = cv2.imread(file_path)
                images.append(img)

                
                bboxes = _load_bbox(os.path.join(bbox_dir_path, dir_name, file_name + ".bboxes.txt"))
                image_logos = []
                for bbox in bboxes:
                    logo_bbox = LogoBBox(
                        logo_id = brand_to_index[dir_name],
                        x=bbox[0],
                        y=bbox[1],
                        w=bbox[2],
                        h=bbox[3])
                    image_logos.append(logo_bbox)
                logos.append(image_logos)
    return images, logos

def load_trainset(dataset_root_path):
    return load(dataset_root_path, "40_images_per_class_train.txt")

def load_testset(dataset_root_path):
    return load(dataset_root_path, "30_images_per_class_test.txt")

if __name__=="__main__":
    load_trainset('./data')