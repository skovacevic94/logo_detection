import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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
    df_cm = pd.DataFrame(cmatrix, range(10), range(10))
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
    print("Mean recal")
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
            img = cv2.equalizeHist(img)
            images.append(img)

            image_logos = load_logos(bbox_dir_path, dir_name, file_name)
            logos.append(image_logos)
            classes.append(image_logos[0].logo_idx)
    return train_test_split(images, logos, test_size=test_size, stratify=classes, random_state=42)


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


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    images, _, logos, _ = load_data('./data')

    classes = []
    areas = []
    aspectRatios = []
    for logo_bboxes in logos:
        for logo_bbox in logo_bboxes:
            id = logo_bbox.logo_id
            w = logo_bbox.w
            h = logo_bbox.h
            areas.append(w*h)
            aspectRatios.append(h/w)
            classes.append(id)
    X = np.zeros((len(areas), 2))
    X[:, 0] = np.array(areas)
    X[:, 1] = np.array(aspectRatios)

    scaler = StandardScaler()

    X_transformed = scaler.fit_transform(X)
    
    clf = KMeans(n_clusters=6).fit(X_transformed)
    scaled_centroids = clf.cluster_centers_
    unscaled_centroids = scaler.inverse_transform(scaled_centroids)
    
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(unscaled_centroids[:, 0], unscaled_centroids[:, 1], s=80)
    plt.legend()
    plt.show()
    print(unscaled_centroids)