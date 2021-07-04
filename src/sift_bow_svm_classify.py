from utils import load_trainset, load_testset, BoundingBox, compute_metrics, plot_confusion_matrix
from tqdm import tqdm
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

if __name__=='__main__':
    sift = cv2.SIFT_create()

    train_images, train_logos = load_trainset('./data')
    
    y_train = np.zeros(len(train_images))   
    train_image_descriptors = []
    for i, img in enumerate(tqdm(iterable=train_images, desc="Computing SIFT descriptors")):
        bbox = train_logos[i][0]
        img = img[bbox.y:bbox.y+bbox.h, bbox.x:bbox.x+bbox.w]
        _, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            continue
        train_image_descriptors.append(descriptors)

        y_train[i] = train_logos[i][0].logo_idx

    k = 130

    print(f"Computing vocabulary of size {k}...")
    all_descriptors = train_image_descriptors[0]
    for descriptor in train_image_descriptors[1:]:
        all_descriptors = np.vstack((all_descriptors, descriptor))
    voc, variance = kmeans(all_descriptors, k, iter=5)

    X_train = np.zeros((len(train_images), k))
    for i, desc in enumerate(tqdm(iterable=train_image_descriptors, desc="Computing features (Quantization + Histogram)")):
        words,distance = vq(desc, voc)
        for w in words:
            X_train[i,w] += 1

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    clf = LinearSVC(penalty='l1', dual=False, C=0.5)
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
    
    y_test = np.zeros(len(test_images))   
    test_image_descriptors = []
    for i, img in enumerate(tqdm(iterable=test_images, desc="Computing SIFT descriptors")):
        bbox = test_logos[i][0]
        img = img[bbox.y:bbox.y+bbox.h, bbox.x:bbox.x+bbox.w]
        _, descriptors = sift.detectAndCompute(img, None)
        test_image_descriptors.append(descriptors)

        y_test[i] = test_logos[i][0].logo_idx

    X_test = np.zeros((len(test_images), k))
    for i, desc in enumerate(tqdm(iterable=test_image_descriptors, desc="Computing features (Quantization + Histogram)")):
        words,distance = vq(desc, voc)
        for w in words:
            X_test[i,w] += 1

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


