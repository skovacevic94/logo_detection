from random import Random
from utils import load_data, transform_to_classification_dataset, report_metrics
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation = inter)


def prepare_size(image, dim, keep_ratio=True, inter = cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if not keep_ratio or w == h:
        return cv2.resize(image, (dim, dim), interpolation=inter)

    if w > h:
        resized = image_resize(image, width=dim, inter=inter)
    else:
        resized = image_resize(image, height=dim, inter=inter)
    res = np.zeros((dim, dim), dtype=np.uint8)
    res[0:resized.shape[0], 0:resized.shape[1]] = resized
    return res


def compute_features(images):
    features = None
    for i, img in enumerate(tqdm(iterable=images, desc="Computing features (Quantization + Histogram)")):
        dim = 64
        img = prepare_size(img, dim, keep_ratio=False, inter=cv2.INTER_LINEAR)
        H = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        if features is None:
            features = H
        else:
            features = np.vstack((features, H))
    return features

if __name__=='__main__':
    np.random.seed(42)
    recompute_vocabulary = False

    clf1 = LinearSVC(C=0.0005)
    clf2 = LinearSVC(C=0.0005)
    
    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    train_images, y_train = transform_to_classification_dataset(train_images, train_logos)
    test_images, y_test = transform_to_classification_dataset(test_images, test_logos)

    y_train_step1 = np.zeros(len(y_train), np.uint)
    for i in range(len(y_train_step1)):
        if y_train[i] == 10:
            y_train_step1[i] = 0
        else:
            y_train_step1[i] = 1
    
    y_test_step1 = np.zeros(len(y_test), np.uint)
    for i in range(len(y_test_step1)):
        if y_test[i] == 10:
            y_test_step1[i] = 0
        else:
            y_test_step1[i] = 1
    
    X_train = compute_features(train_images)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    print("Training stage step 1...")
    clf1.fit(X_train, y_train_step1)
    y_pred_step1 = clf1.predict(X_train)
    report_metrics(y_train_step1, y_pred_step1, "Train step 1")
    
    print("Training stage step 2...")
    pos_idx = np.where(y_train_step1==1)
    clf2.fit(X_train[pos_idx], y_train[pos_idx])
    y_pred_step2 = clf2.predict(X_train[pos_idx])
    report_metrics(y_train[pos_idx], y_pred_step2, "Train step 2")

    X_test = compute_features(test_images)
    X_test = scaler.transform(X_test)
    
    print("Evaluation stage step 1...")
    y_pred_step1 = clf1.predict(X_test)
    report_metrics(y_test_step1, y_pred_step1, "Test step 1")

    print("Evaluation stage step 2...")
    pos_idx = np.where(y_test_step1==1)
    y_pred_step2 = clf2.predict(X_test[pos_idx])
    report_metrics(y_test[pos_idx], y_pred_step2, "Test step 2")

    print("Evaluation combined")
    y_pred_combined = np.ones(len(y_test))*10
    y_pred_combined[pos_idx] = y_pred_step2
    report_metrics(y_test, y_pred_combined, "Test combined")
    

