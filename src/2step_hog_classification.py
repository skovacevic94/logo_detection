from utils import load_data, transform_to_classification_dataset, report_metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from features import compute_hog_features


if __name__=='__main__':
    np.random.seed(42)

    clf1 = LinearSVC(C=0.0005)
    clf2 = LinearSVC(C=0.0005)
    #clf1 = RandomForestClassifier(n_estimators=1000)

    train_images, test_images, train_logos, test_logos = load_data('./data', test_size=0.33)
    train_images, y_train = transform_to_classification_dataset(train_images, train_logos)
    test_images, y_test = transform_to_classification_dataset(test_images, test_logos)

    X_train = compute_hog_features(train_images)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = compute_hog_features(test_images)
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
    

