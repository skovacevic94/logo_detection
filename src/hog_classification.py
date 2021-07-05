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
        H = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        if features is None:
            features = H
        else:
            features = np.vstack((features, H))
    return features

if __name__=='__main__':
    np.random.seed(42)
    recompute_vocabulary = False

    clf = LinearSVC(C=0.001)

    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    train_images, y_train = transform_to_classification_dataset(train_images, train_logos)
    test_images, y_test = transform_to_classification_dataset(test_images, test_logos)
    
    print("Training stage...")
    X_train = compute_features(train_images)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    report_metrics(y_train, y_pred, "Train")

    print("Evaluation stage...")
    X_test = compute_features(test_images)
    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)
    report_metrics(y_test, y_pred, "Test")
    

