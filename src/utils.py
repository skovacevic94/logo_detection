import os
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
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    images, logos = load_trainset('./data')

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