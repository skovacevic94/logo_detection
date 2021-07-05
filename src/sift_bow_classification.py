from random import Random
from utils import load_data, transform_to_classification_dataset, report_metrics
from tqdm import tqdm
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import os

def create_vocabulary(images, sift, k=130):
    all_descriptors_list = []
    print(f"Computing vocabulary of size {k}...")
    for img in tqdm(iterable=images, desc="Computing SIFT descriptors"):
        _, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            continue
        all_descriptors_list.append(descriptors)

    all_descriptors = all_descriptors_list[0]
    for descriptor in all_descriptors_list[1:]:
        all_descriptors = np.vstack((all_descriptors, descriptor))
    voc, variance = kmeans(all_descriptors, k, iter=2)
    return voc

def compute_features(images, sift, voc):
    features = np.zeros((len(images), len(voc)))
    for i, img in enumerate(tqdm(iterable=images, desc="Computing features (Quantization + Histogram)")):
        _, desc = sift.detectAndCompute(img, None)
        if desc is None:
            continue
        words,distance = vq(desc, voc)
        for w in words:
            features[i,w] += 1
    return features

if __name__=='__main__':
    np.random.seed(42)
    recompute_vocabulary = False

    sift = cv2.SIFT_create()
    clf = RandomForestClassifier(n_estimators=1500, max_features=20, class_weight='balanced')

    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    train_images, y_train = transform_to_classification_dataset(train_images, train_logos)
    test_images, y_test = transform_to_classification_dataset(test_images, test_logos)
    
    k = 3000
    voc_filename = f"voc_k{k}.pkl"
    if os.path.exists(voc_filename):
        print("Using cached vocabulary!")
        with open(voc_filename, "rb") as voc_file:
            voc = pkl.load(voc_file)
    else:
        voc = create_vocabulary(train_images, sift, k)
        with open(voc_filename, "wb") as voc_file:
            pkl.dump(voc, voc_file)
        
    print("Training stage...")
    X_train = compute_features(train_images, sift, voc)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    report_metrics(y_train, y_pred, "Train")

    print("Evaluation stage...")
    X_test = compute_features(test_images, sift, voc)
    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)
    report_metrics(y_test, y_pred, "Test")
    

