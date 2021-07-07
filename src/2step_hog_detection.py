from utils import load_data, transform_to_classification_dataset, report_metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from features import compute_hog_features
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import cv2

if __name__=='__main__':
    np.random.seed(42)

    cv2.ocl.setUseOpenCL(True)
    print(
        "INFO: OpenCL - available: ",
        cv2.ocl.haveOpenCL(),
        " using: ",
        cv2.ocl.useOpenCL())

    #clf1 = LinearSVC(C=0.0005)
    clf2 = LinearSVC(C=0.0005)
    clf1 = RandomForestClassifier(verbose=1, max_features=10, n_estimators=2000, n_jobs=10)

    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    train_images_clf, y_train = transform_to_classification_dataset(train_images, train_logos)
    test_images_clf, y_test = transform_to_classification_dataset(test_images, test_logos)

    X_train = compute_hog_features(train_images_clf)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = compute_hog_features(test_images_clf)
    X_test = scaler.transform(X_test)
    
    s_train = np.zeros(len(y_train), np.uint)
    for i in range(len(s_train)):
        if y_train[i] == 10:
            s_train[i] = 0
        else:
            s_train[i] = 1
    
    s_test = np.zeros(len(y_test), np.uint)
    for i in range(len(s_test)):
        if y_test[i] == 10:
            s_test[i] = 0
        else:
            s_test[i] = 1
    
    print("Training Logo/No-Logo classifier")
    clf1.fit(X_train, s_train)
    s_pred = clf1.predict(X_train)
    report_metrics(s_train, s_pred, "Logo/No-Logo classifier on trainset")
    
    print("Training Brand classifier")
    logo_idx = np.where(s_train==1)
    p_train = y_train[logo_idx]
    clf2.fit(X_train[logo_idx], p_train)
    p_pred = clf2.predict(X_train[logo_idx])
    report_metrics(p_train, p_pred, "Barnd classifier on trainset")

    print("Evaluation Logo/No-Logo classifier...")
    s_pred = clf1.predict(X_test)
    report_metrics(s_test, s_pred, "Logo/No-logo classifier on testset")

    print("Evaluation Brand classifier...")
    logo_idx = np.where(s_test==1)
    p_test = y_test[logo_idx]
    p_pred = clf2.predict(X_test[logo_idx])
    report_metrics(p_test, p_pred, "Brand classifier on testset")

    print("Evaluation full...")
    logo_idx = np.where(s_pred==1)
    p_pred = clf2.predict(X_test[logo_idx])
    y_pred = np.ones(len(y_test))*10
    y_pred[np.where(s_pred==1)] = p_pred
    report_metrics(y_test, y_pred, "Testset")
    
    print("Performing sliding window detection")
    stride = 5
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
        X = compute_hog_features(windows)
        X = scaler.transform(X)
        #pred = clf1.predict(X)
        probs = clf1.predict_proba(X)
        detected_idx = np.argwhere(probs > 0.7)
        rec_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for idx in detected_idx:
            idx = idx[0]
            cv2.rectangle(rec_img, (xv[idx], yv[idx]), (xv[idx]+window_size[1], yv[idx]+window_size[0]), (0, 255, 0))
        cv2.imshow("RESULt", rec_img)
        cv2.waitKey(0)