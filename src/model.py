import abc
import numpy as np
from tqdm import tqdm
import cv2
from typing import List
from dataclasses import dataclass





class Model(abc.ABC):
    @abc.abstractmethod
    def train(self, images: list, indices: list, logo_labels: list):
        pass

    @abc.abstractmethod
    def detect(self, images: list) -> List[LogoBBox]:
        pass


class FeatureMatcher(Model):
    def __init__(self, index_params, search_params):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def train(self, images: list, indices: list, logo_labels: list):
        progress_bar = tqdm(total=len(indices), desc="Computing features")
        self.logo_ids = []
        for i, img_idx in enumerate(indices):
            img = images[img_idx]
            logo_id, bbox = logo_labels[i]

            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            logo_img = img[y:y+h, x:x+w]

            _, descriptors = self.sift.detectAndCompute(logo_img, None)
            self.flann.add([descriptors])
            self.logo_ids.append(logo_id)
            progress_bar.update()
        self.flann.train()


    def detect(self, images: list) -> List[LogoBBox]:
        output = []
        keypoints = self.sift.detect(images)
        keypoints, descriptors = self.sift.compute(images, keypoints)
        progress_bar = tqdm(total=len(images), desc="Logo detection")
        for i, desc in enumerate(descriptors):
            matches = self.flann.knnMatch(desc, 5)
            for match in matches:
                if match[0].distance < 0.7*match[1].distance:
                    
            matchesMask = [[0,0] for i in range(len(matches))]

            progress_bar.update()
        return output

class HOGSVM(Model):
    def train(self, images: list, indices: list, logo_labels: list):
        pass

    def detect(self, images: list) -> List[LogoBBox]:
        pass