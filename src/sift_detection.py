from utils import load_data, transform_to_classification_dataset, report_metrics, get_window_sizes
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import os
from features import create_vocabulary, compute_sift_bow_features
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


if __name__=='__main__':
    np.random.seed(42)

    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    train_images_clf, y_train = transform_to_classification_dataset(train_images, train_logos)
    test_images_clf, y_test = transform_to_classification_dataset(test_images, test_logos)
    
    clf = RandomForestClassifier(n_estimators=1500, max_features=20, class_weight='balanced')

    k = 3000
    voc_filename = f"voc_k{k}.pkl"
    if os.path.exists(voc_filename):
        print("Using cached vocabulary!")
        with open(voc_filename, "rb") as voc_file:
            voc = pkl.load(voc_file)
    else:
        voc = create_vocabulary(train_images_clf, k)
        with open(voc_filename, "wb") as voc_file:
            pkl.dump(voc, voc_file)
        
    print("Training stage...")
    X_train = compute_sift_bow_features(train_images_clf, voc)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    report_metrics(y_train, y_pred, "Train")

    print("Evaluation stage...")
    X_test = compute_sift_bow_features(test_images_clf, voc)
    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)
    report_metrics(y_test, y_pred, "Test")
    
    s_test = np.ones(len(y_test), np.uint)
    nologo_idx = np.where(y_test == 10)
    s_test[nologo_idx] = 0
    
    s_pred = np.ones(len(y_pred), np.uint)
    nologo_idx = np.where(y_pred == 10)
    s_pred[nologo_idx] = 0
    report_metrics(s_test, s_pred, "Logo/No-Logo results on testset")

    print("Performing sliding window detection")
    stride = 1
    for i, image in enumerate(tqdm(test_images)):
        img_h, img_w = image.shape
        logo = test_logos[i][0]
        window_size = (logo.h, logo.w)
        windows = sliding_window_view(image, window_size)[::stride, ::stride, :, :]
        xs = np.arange(img_w-window_size[1]+1, step=stride)
        ys = np.arange(img_h-window_size[0]+1, step=stride)
        xv, yv = np.meshgrid(xs, ys)
        xv = np.reshape(xv, -1)
        yv = np.reshape(yv, -1)
        assert(len(xs)==windows.shape[1])
        assert(len(ys)==windows.shape[0])
        windows = np.reshape(windows, (windows.shape[0]*windows.shape[1], windows.shape[2], windows.shape[3]))
        detected = False
        X = compute_sift_bow_features(windows, voc)
        X = scaler.transform(X)
        pred = clf.predict(X)
        detected_idx = np.argwhere(pred != 10)
        rec_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for idx in detected_idx:
            idx = idx[0]
            cv2.rectangle(rec_img, (xv[idx], yv[idx]), (xv[idx]+window_size[1], yv[idx]+window_size[0]), (0, 255, 0))
        cv2.imshow("RESULt", rec_img)
        cv2.waitKey(0)
        print(f"Detected {len(pred[pred != 10])} Exists {len(test_logos[i])}")

