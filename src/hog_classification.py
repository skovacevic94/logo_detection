from random import Random
from utils import load_data, transform_to_classification_dataset, report_metrics
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog


def compute_features(images):
    features = None
    for i, img in enumerate(tqdm(iterable=images, desc="Computing features (Quantization + Histogram)")):
        h, w = img.shape

        img = cv2.resize(img, (64, 64))
        H = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
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
    

