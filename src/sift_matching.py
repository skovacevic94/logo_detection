from utils import load_trainset, load_testset, LogoBBox
from tqdm import tqdm
import cv2
import numpy as np

def compute_metrics(true_logos, detected_logos):
    assert(len(true_logos)==len(detected_logos))

    confussion_matrix = np.zeros((11, 11))
    for i in range(len(true_logos)):
        logo_index = true_logos[i][0].logo_id
        to_detect = 1 #len(true_logos[i]) # Number of logos that algorithms should detect on the image
        for detected_logo_bbox in detected_logos[i]:
            confussion_matrix[detected_logo_bbox.logo_id][logo_index] += 1
            to_detect -= 1
        confussion_matrix[10][logo_index] += to_detect
    
    precission = np.zeros(10)
    recall = np.zeros(10)
    for i in range(10):
        precission[i] = confussion_matrix[i][i] / np.sum(confussion_matrix[i, :])
        recall[i] = confussion_matrix[i][i] / np.sum(confussion_matrix[:, i])
    accuracy = np.trace(confussion_matrix)/np.sum(confussion_matrix)
    return confussion_matrix, precission, recall, accuracy

def test_and_evaluate(sift, flann, intraclass_detectors, logo_ids):
    images, logos = load_testset("./data")
    
    keypoints = sift.detect(images)
    keypoints, descriptors = sift.compute(images, keypoints)
    
    detected_logos = []
    progress_bar = tqdm(total=len(images), desc="Testing")
    for i, desc in enumerate(descriptors):
        matches = flann.knnMatch(desc, 5)
        votes = np.zeros(10)
        for match in matches:
            if match[0].distance < 0.85*match[1].distance:
                votes[logo_ids[match[0].imgIdx]] += 1
        detected_logo_index = 10
        x, y, w, h =0, 0, 0, 0
        if np.max(votes) >= 4:
            detected_logo_index = np.argmax(votes)
            best_match_dist = np.inf
            best_match_img_idx = -1
            best_good_matches = None
            for trainImgIdx, intraclass_flann in enumerate(intraclass_detectors[detected_logo_index]):
                specific_image_matches = intraclass_flann.knnMatch(desc, 5)
                good_matches = []
                for match in matches:
                    if match[0].distance < 0.85*match[1].distance:
                        good_matches.append(match[0])
                m = 0
                for match in good_matches:
                    m += match.distance
                m /= len(good_matches)
                if m < best_match_dist and len(good_matches) > 10:
                    best_match_dist = m
                    best_match_img_idx = trainImgIdx
                    best_good_matches = good_matches
        
        detected_logos.append([LogoBBox(detected_logo_index, x, y, w, h)])
        progress_bar.update()

    confussion_matrix, precission, recall, accuracy = compute_metrics(logos, detected_logos)
    print(confussion_matrix)
    print(np.mean(precission))
    print(np.mean(recall))
    print(accuracy)
    return detected_logos


if __name__=='__main__':
    sift = cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    images, logos = load_trainset('./data')
    logo_ids = []
    progress_bar = tqdm(total=len(images), desc="Training")
    intraclass_detectors = {}
    for i in range(len(images)):
        img = images[i]
        for logo_bbox in logos[i]:
            x, y, w, h = logo_bbox.x, logo_bbox.y, logo_bbox.w, logo_bbox.h
            
            logo_img = img[y:y+h, x:x+w]
            _, descriptors = sift.detectAndCompute(logo_img, None)
            if descriptors is None:
                continue
            flann.add([descriptors])
            logo_ids.append(logo_bbox.logo_id)
            if logo_bbox.logo_id not in intraclass_detectors:
                intraclass_detectors[logo_bbox.logo_id] = []
            specific_flann = cv2.FlannBasedMatcher(index_params, search_params)
            specific_flann.add([descriptors])
            specific_flann.train()
            intraclass_detectors[logo_bbox.logo_id].append(specific_flann)
        progress_bar.update()
    flann.train()

    test_and_evaluate(sift, flann, intraclass_detectors, logo_ids)


