from utils import load_data, report_metrics
from tqdm import tqdm
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def create_vocabulary(sift, images, k=130, logos=None):
    all_descriptors_list = []
    for i, img in enumerate(tqdm(iterable=images, desc="Computing SIFT descriptors")):
        if logos is not None:
            bbox = logos[i][0]
            img = img[bbox.y:bbox.y+bbox.h, bbox.x:bbox.x+bbox.w]
        _, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            continue
        all_descriptors_list.append(descriptors)

    print(f"Computing vocabulary of size {k}...")
    all_descriptors = all_descriptors_list[0]
    for descriptor in all_descriptors_list[1:]:
        all_descriptors = np.vstack((all_descriptors, descriptor))
    voc, variance = kmeans(all_descriptors, k, iter=1)
    return voc

def compute_features(sift, voc, images, logos):
    X = np.zeros((len(images), len(voc)))
    y = np.zeros(len(images))
    for i, img in enumerate(tqdm(iterable=images, desc="Computing features (Quantization + Histogram)")):
        bbox = logos[i][0]
        img = img[bbox.y:bbox.y+bbox.h, bbox.x:bbox.x+bbox.w]
        _, desc = sift.detectAndCompute(img, None)
        if desc is None:
            continue
        words,distance = vq(desc, voc)
        for w in words:
            X[i,w] = 1
        y[i] = logos[i][0].logo_idx
    return X, y

if __name__=='__main__':
    sift = cv2.SIFT_create()
    clf = LinearSVC(C=0.01)
    
    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    
    k = 3000
    voc = create_vocabulary(sift, train_images, k, train_logos)

    print("Training stage...")
    X_train, y_train = compute_features(sift, voc, train_images, train_logos)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    report_metrics(y_train, y_pred, "Train")

    print("Evaluation stage...")
    X_test, y_test = compute_features(sift, voc, test_images, test_logos)
    y_pred = clf.predict(X_test)
    report_metrics(y_test, y_pred, "Test")
    

