from utils import load_data, transform_to_classification_dataset, report_metrics, get_window_sizes
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import pickle as pkl
import os
from features import create_vocabulary, compute_sift_bow_features
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


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


if __name__=='__main__':
    np.random.seed(42)

    cv2.ocl.setUseOpenCL(True)
    print(
        "INFO: OpenCL - available: ",
        cv2.ocl.haveOpenCL(),
        " using: ",
        cv2.ocl.useOpenCL())

    stride = 5

    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    
    k = 3000
    voc_filename = f"voc_k{k}.pkl"
    with open(voc_filename, "rb") as voc_file:
        voc = pkl.load(voc_file)
    
    window_sizes = [((64, 64), 1.0)]
    clfs = []
    scalers = []
    for window_params in window_sizes:
        window_size = window_params[0]
        scale_factor = window_params[1]
        print("Training stage...")
        train_images_clf, y_train = transform_to_classification_dataset(train_images, train_logos, window_size, scale_factor)
        X_train = compute_sift_bow_features(train_images_clf, voc)
        scalers.append(StandardScaler().fit(X_train))
        X_train = scalers[-1].transform(X_train)
        clfs.append(RandomForestClassifier(n_estimators=1500, max_features=7, n_jobs=10, min_samples_split=4))
        clfs[-1].fit(X_train, y_train)
        y_pred = clfs[-1].predict(X_train)
        report_metrics(y_train, y_pred, "Train")

        print("Evaluation stage...")
        test_images_clf, y_test = transform_to_classification_dataset(test_images, test_logos, window_size, scale_factor)
        X_test = compute_sift_bow_features(test_images_clf, voc)
        X_test = scalers[-1].transform(X_test)
        y_pred = clfs[-1].predict(X_test)
        report_metrics(y_test, y_pred, "Test")

        s_test = np.ones(len(y_test), np.uint)
        nologo_idx = np.where(y_test == 10)
        s_test[nologo_idx] = 0
        
        s_pred = np.ones(len(y_pred), np.uint)
        nologo_idx = np.where(y_pred == 10)
        s_pred[nologo_idx] = 0
        report_metrics(s_test, s_pred, "Logo/No-Logo results on testset")

    print("Performing sliding window detection")
    for c, window_params in enumerate(window_sizes):
        window_size = window_params[0]
        scale_factor = window_params[1]
        for i, image in enumerate(tqdm(test_images[:3])):
            image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
            img_h, img_w = image.shape
            logo = test_logos[i][0]
            #window_size = (logo.h, logo.w)
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
            X = scalers[c].transform(X)
            pred = clfs[c].predict(X)
            detected_idx = np.argwhere(pred != 10)
            rec_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for idx in detected_idx:
                idx = idx[0]
                cv2.rectangle(rec_img, (xv[idx], yv[idx]), (xv[idx]+window_size[1], yv[idx]+window_size[0]), (0, 255, 0))
            cv2.imshow("RESULT", rec_img)
            cv2.waitKey(0)
            print(f"Detected {len(pred[pred != 10])} Exists {len(test_logos[i])}")

