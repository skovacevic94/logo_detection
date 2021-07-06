import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    "supreme": 9,
    "None": 10
}

index_to_brand = {}
for k, v in brand_to_index.items():
    index_to_brand[v] = k


@dataclass
class BoundingBox:
    logo_idx: int
    x: int
    y: int
    w: int
    h: int


def report_metrics(y_true, y_pred, title):
    cmatrix = confusion_matrix(y_true, y_pred)
    n_classes = np.max(y_true) + 1
    df_cm = pd.DataFrame(cmatrix, range(n_classes), range(n_classes))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.title(title)
    plt.show()

    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred)
    print("Per-class precision")
    print(precision)
    print("Per-class recall")
    print(recall)
    print("Mean precision")
    print(np.mean(precision))
    print("Mean recall")
    print(np.mean(recall))


def load_logos(bbox_dir_path, dir_name, file_name):
    image_logos = []
    with open(os.path.join(bbox_dir_path, dir_name, file_name + ".bboxes.txt"), "r") as bbox_file:
        lines = bbox_file.readlines()
        for line in lines:
            x, y, w, h = [int(num) for num in line.split()]
            image_logos.append(BoundingBox(brand_to_index[dir_name], x, y, w, h))
    return image_logos


def load_data(dataset_root_path, test_size):
    bbox_dir_path = os.path.join(dataset_root_path, "masks")
    image_root_dir_path = os.path.join(dataset_root_path, "jpg")

    images = []
    logos = []
    classes = []
    for dir_name in os.listdir(image_root_dir_path):
        dir_path = os.path.join(image_root_dir_path, dir_name)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            img = cv2.imread(file_path)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            images.append(img)

            image_logos = load_logos(bbox_dir_path, dir_name, file_name)
            logos.append(image_logos)
            classes.append(image_logos[0].logo_idx)
    return train_test_split(images, logos, test_size=test_size, stratify=classes, random_state=42)


def rect_overlap(bbox1, bbox2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bbox1.x, bbox2.x)
    yA = max(bbox1.y, bbox2.y)
    xB = min(bbox1.x+bbox1.w, bbox2.x+bbox2.w)
    yB = min(bbox1.y+bbox1.h, bbox2.y+bbox2.h)

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return False
    return True


def transform_to_classification_dataset(images, logos, include_negatives = True):
    labels = []
    data = []
    for i, img in enumerate(images):
        bbox = logos[i][0]
        img_h, img_w = img.shape
        # Get positive example
        pos_img = img[bbox.y:bbox.y+bbox.h, bbox.x:bbox.x+bbox.w]
        data.append(pos_img)
        labels.append(logos[i][0].logo_idx)
        # Generate negative example
        for i in range(10): # Try 10 times
            w, h = bbox.w, bbox.h

            x = np.random.randint(0, img_w - w)
            y = np.random.randint(0, img_h - h)

            candidate_bbox = BoundingBox(bbox.logo_idx, x, y, w, h)
            valid = True
            for positive_bbox in logos[i]:
                if rect_overlap(candidate_bbox, positive_bbox):
                    valid = False
                    break
            if valid:
                neg_img = img[y:y+h, x:x+w]
                data.append(neg_img)
                labels.append(10) # None-detected class 
                break
    return data, np.array(labels)


def compute_metrics(true_logos, detected_logos):
    assert(len(true_logos)==len(detected_logos))

    confussion_matrix = np.zeros((11, 11))
    for i in range(len(true_logos)):
        logo_index = true_logos[i][0].logo_id
        to_detect = len(true_logos[i]) # Number of logos that algorithms should detect on the image
        for detected_logo_bbox in detected_logos[i]:
            confussion_matrix[detected_logo_bbox.logo_id][logo_index] += 1
            to_detect -= 1
        confussion_matrix[10][logo_index] += to_detect
    
    precission = np.zeros(10)
    recall = np.zeros(10)
    for i in range(10):
        precission[i] = confussion_matrix[i][i] / np.sum(confussion_matrix[i, :])
        recall[i] = confussion_matrix[i][i] / np.sum(confussion_matrix[:, i])
    accuracy = np.trace(confussion_matrix)/np.sum(confussion_matrix)
    return confussion_matrix, precission, recall, accuracy


def get_window_sizes(logos, k):
    classes = []
    widths = []
    heights = []
    for logo_bboxes in logos:
        for logo_bbox in logo_bboxes:
            id = logo_bbox.logo_idx
            widths.append(logo_bbox.w)
            heights.append(logo_bbox.h)
            classes.append(id)
    X = np.zeros((len(widths), 2))
    X[:, 0] = np.array(widths)
    X[:, 1] = np.array(heights)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = KMeans(n_clusters=5).fit(X)
    centroids = clf.cluster_centers_

    centroids = scaler.inverse_transform(centroids)
    X = scaler.inverse_transform(X)
    
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80)
    plt.legend()
    plt.title("Windowing clusters")
    plt.show()

    return [(256, 256)]
    #return centroids.astype(np.int)

    