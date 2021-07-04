from utils import load_data, report_metrics
from tqdm import tqdm
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def create_vocabulary(images, logos, sift, k=130, roi=False):
    all_descriptors_list = []
    for i, img in enumerate(tqdm(iterable=images, desc="Computing SIFT descriptors")):
        if roi:
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

def compute_features(images, logos, sift, voc, roi=False):
    X = np.zeros((len(images), len(voc)))
    y = np.zeros(len(images))
    for i, img in enumerate(tqdm(iterable=images, desc="Computing features (Quantization + Histogram)")):
        if roi:
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
    voc_split = None # 0.4

    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    voc_images = train_images
    voc_logos = train_logos
    svm_images = train_images
    svm_logos = train_logos
    if voc_split is not None:
        voc_images, svm_images, voc_logos, svm_logos = train_test_split(train_images, train_logos, train_size=voc_split, random_state=42)
    
    k = 3000
    voc = create_vocabulary(voc_images, voc_logos, sift, k, False)

    print("Training stage...")
    X_train, y_train = compute_features(svm_images, svm_logos, sift, voc, False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    report_metrics(y_train, y_pred, "Train")

    print("Evaluation stage...")
    X_test, y_test = compute_features(test_images, test_logos, sift, voc, False)
    y_pred = clf.predict(X_test)
    report_metrics(y_test, y_pred, "Test")
    

