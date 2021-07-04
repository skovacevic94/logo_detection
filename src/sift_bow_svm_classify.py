from utils import load_trainset, load_testset, BoundingBox, compute_metrics, plot_confusion_matrix
from tqdm import tqdm
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

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
            X[i,w] += 1
        y[i] = logos[i][0].logo_idx
    return X, y

if __name__=='__main__':
    sift = cv2.SIFT_create()

    train_images, train_logos = load_trainset('./data')
    
    k = 130
    voc = create_vocabulary(sift, train_images, k, train_logos)
    X_train, y_train = compute_features(sift, voc, train_images, train_logos)
    
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    clf = LinearSVC(C=0.5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)

    plot_confusion_matrix(y_train, y_pred, "Train")
    precision, recall, _, _ = precision_recall_fscore_support(y_train, y_pred)
    print("Per-class precision")
    print(precision)
    print("Per-class recall")
    print(recall)
    print("Mean precision")
    print(np.mean(precision))
    print("Mean recal")
    print(np.mean(recall))

    print("Evaluation stage...")
    test_images, test_logos = load_testset("./data")

    X_test, y_test = compute_features(sift, voc, test_images, test_logos)

    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)

    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, "Test")
    print("Per-class precision")
    print(precision)
    print("Per-class recall")
    print(recall)
    print("Mean precision")
    print(np.mean(precision))
    print("Mean recal")
    print(np.mean(recall))
    #test_and_evaluate(sift, y_true)


