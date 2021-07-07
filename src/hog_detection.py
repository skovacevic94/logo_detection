from features import compute_hog_features, create_vocabulary, compute_sift_bow_features
from utils import BoundingBox, load_data, transform_to_classification_dataset, report_metrics, index_to_brand, get_window_sizes
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import cv2
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import pickle as pkl
import os


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

    stride = 20
    clf = LinearSVC(C=0.0001)


    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)

    k = 800
    voc_filename = f"voc_k{k}.pkl"
    if os.path.exists(voc_filename):
        print("Using cached vocabulary!")
        with open(voc_filename, "rb") as voc_file:
            voc = pkl.load(voc_file)
    else:
        voc = create_vocabulary(train_images, k)
        with open(voc_filename, "wb") as voc_file:
            pkl.dump(voc, voc_file)

    
    print("Training stage...")
    train_images_clf, y_train = transform_to_classification_dataset(train_images, train_logos, stride)
    y_train_det = np.ones(len(y_train), np.uint8)
    y_train_det[y_train==10] = 0

    X_train = np.hstack((compute_hog_features(train_images_clf), compute_sift_bow_features(train_images_clf, voc)))
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    
    clf.fit(X_train, y_train_det)
    y_pred = clf.predict(X_train)
    report_metrics(y_train_det, y_pred, "Train")

    print("Evaluation stage...")
    test_images_clf, y_test = transform_to_classification_dataset(test_images, test_logos, stride)
    y_test_det = np.ones(len(y_test), np.uint8)
    y_test_det[y_test==10] = 0
    X_test = np.hstack((compute_hog_features(test_images_clf), compute_sift_bow_features(test_images_clf, voc)))
    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)
    report_metrics(y_test_det, y_pred, "Test")

    print("Performing sliding window detection")
    for i, image in enumerate(tqdm(train_images[:3])):
        img_h, img_w = image.shape
        logo = train_logos[i][0]
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
        X = np.hstack((compute_hog_features(windows), compute_sift_bow_features(windows, voc)))
        X = scaler.transform(X)
        pred = clf.predict(X)
        distances = clf.decision_function(X)
        detected_idx = np.reshape(np.argwhere(pred == 1), -1)
        top_dist_incices = np.argsort(np.reshape(distances[detected_idx], -1))[:20]
        rec_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for idx in detected_idx:
            cv2.rectangle(rec_img, (xv[idx], yv[idx]), (xv[idx]+window_size[1], yv[idx]+window_size[0]), (0, 255, 0))
        cv2.imshow("RESULt", rec_img)
        cv2.waitKey(0)
        print(f"Detected {len(pred[pred != 10])} Exists {len(test_logos[i])}")


    
    

