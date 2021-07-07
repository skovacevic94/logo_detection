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
    n_classes = np.max(y_true) + 1
    cmatrix = confusion_matrix(y_true, y_pred, labels=range(n_classes))
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


def iou(bbox1, bbox2):
    xA = max(bbox1.x, bbox2.x)
    yA = max(bbox1.y, bbox2.y)
    xB = min(bbox1.x+bbox1.w, bbox2.x+bbox2.w)
    yB = min(bbox1.y+bbox1.h, bbox2.y+bbox2.h)

    interArea = abs(max((xB - xA), 0) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    boxAArea = bbox1.w*bbox1.h
    boxBArea = bbox2.w*bbox2.h

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def io2(bbox1, bbox2):
    xA = max(bbox1.x, bbox2.x)
    yA = max(bbox1.y, bbox2.y)
    xB = min(bbox1.x+bbox1.w, bbox2.x+bbox2.w)
    yB = min(bbox1.y+bbox1.h, bbox2.y+bbox2.h)

    interArea = abs(max((xB - xA), 0) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    boxAArea = bbox1.w*bbox1.h
    boxBArea = bbox2.w*bbox2.h

    iou = interArea / float(boxBArea)

    return iou


def transform_to_classification_dataset(images, logos, stride):
    labels = []
    data = []
    for i, img in enumerate(images):
        bbox = logos[i][0]
        img_h, img_w = img.shape
        w = bbox.w + 2*stride
        h = bbox.h + 2*stride
        if w > img_w or h > img_h:
            continue
        for _ in range(5):
            true_mid_x = bbox.x + (bbox.w // 2)
            true_mid_y = bbox.y + (bbox.h // 2)

            # Initial proposal
            x = true_mid_x - (w // 2)
            y = true_mid_y - (h // 2)

            # Fix if out of bounds
            if x + w > img_w:
                x -= (x+w-img_w)
            if y + h > img_h:
                y -= (y+h-img_h)

            # Possible x+ perturbation amount
            x_pos_pert = min(img_w-(x+w), abs(bbox.x - x))
            # Possible x- perturbation amount
            x_neg_pert = min(x, abs((x+w)-(bbox.x+bbox.w)))
            # Possible y+ perturbation amount
            y_pos_pert = min(img_h-(y+h), abs(bbox.y - y))
            # Possible y- perturbation amount
            y_neg_pert = min(y, abs((y+h)-(bbox.y+bbox.h)))

            if x_pos_pert > 0 and x_neg_pert:
                x = x+np.random.randint(-x_neg_pert, x_pos_pert)
            if y_pos_pert > 0 and y_neg_pert:
                y = y+np.random.randint(-y_neg_pert, y_pos_pert)

            pos_img = img[y:y+h, x:x+w]
            data.append(pos_img)
            labels.append(logos[i][0].logo_idx)
            #cv2.imshow("POS IMG", pos_img)
            #cv2.resizeWindow("POS IMG", 600, 600)
            #cv2.waitKey(0)
        # Generate negative example
        for _ in range(50):
            for _ in range(5): # Try random 5 times
                x = np.random.randint(0, img_w - w)
                y = np.random.randint(0, img_h - h)

                candidate_bbox = BoundingBox(bbox.logo_idx, x, y, w, h)
                valid = True
                for positive_bbox in logos[i]:
                    if io2(candidate_bbox, positive_bbox) > 0.3:
                        valid = False
                        break
                if valid:
                    neg_img = img[y:y+h, x:x+w]
                    data.append(neg_img)
                    #cv2.imshow("NEG IMG", neg_img)
                    #cv2.resizeWindow("NEG IMG", 600, 600)
                    #cv2.waitKey(0)
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

    return centroids.astype(np.int)
    
if __name__=='__main__':
    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    get_window_sizes(train_logos, 10)