import os
from pathlib import Path
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


def load_set_file(path):
    video_set = set()
    with open(path, 'r') as set_file:
        for line in set_file:
            video_set.add(line.strip())
    return video_set


def load_bbox_file(path):
    bboxes = []
    with open(path, "r") as bbox_file:
         x, y, w, h = [int(num) for num in next(bbox_file).split()]
         bboxes.append([x, y, w, h])
    return bboxes


def load_set(set_file_name):
    root_dir_path = os.path.join(Path(__file__).parent.parent, "project_dataset")
    bbox_dir_path = os.path.join(root_dir_path, "masks")
    image_root_dir_path = os.path.join(root_dir_path, "jpg")

    set_file_path = os.path.join(root_dir_path, "ImageSets", set_file_name)
    trainset = load_set_file(set_file_path)
    
    images = []
    indices = []
    logo_labels = []
    for dir_name in os.listdir(image_root_dir_path):
        dir_path = os.path.join(image_root_dir_path, dir_name)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.splitext(file_name)[0] in trainset:
                img = cv2.imread(file_path)
                images.append(img)

                bboxes = load_bbox_file(os.path.join(bbox_dir_path, dir_name, file_name + ".bboxes.txt"))
                for bbox in bboxes:
                    indices.append(len(images)-1)
                    logo_labels.append([brand_to_int[dir_name], bbox])
    return images, indices, logo_labels


def load_trainset():
    set_file_name = "40_images_per_class_train.txt"
    return load_set(set_file_name)


def load_testset():
    set_file_name = "30_images_per_class_test.txt"
    return load_set(set_file_name)



if __name__=="__main__":
    load_trainset()