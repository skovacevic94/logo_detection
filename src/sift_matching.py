from utils import load_trainset, load_testset
from tqdm import tqdm
import cv2
import numpy as np

def compute_metrics(confussion_matrix):
    precission = np.zeros(10)
    recall = np.zeros(10)

    for i in range(10):
        precission[i] = confussion_matrix[i][i] / np.sum(confussion_matrix[i, :])
        recall[i] = confussion_matrix[i][i] / np.sum(confussion_matrix[:, i])
    accuracy = np.trace(confussion_matrix)/np.sum(confussion_matrix)
    return precission, recall, accuracy

def test_and_evaluate(sift, flann, logo_ids):
    images, logos = load_testset("./data")
    output = []
    keypoints = sift.detect(images)
    keypoints, descriptors = sift.compute(images, keypoints)
    progress_bar = tqdm(total=len(images), desc="Testing")

    confussion_matrix = np.zeros((10, 10))
    for i, desc in enumerate(descriptors):
        matches = flann.knnMatch(desc, 5)
        matchesMask = [[0,0] for i in range(len(matches))]
        votes = np.zeros(10)
        for j, match in enumerate(matches):
            if match[0].distance < 0.85*match[1].distance:
                matchesMask[j] = [1, 0]
                votes[logo_ids[match[0].imgIdx]] += 1
        detected_logo_index = np.argmax(votes)
        confussion_matrix[detected_logo_index][logos[i][0].logo_id] += 1
        progress_bar.update()
    print(confussion_matrix)
    precission, recall, accuracy = compute_metrics(confussion_matrix)
    print(np.mean(precission))
    print(np.mean(recall))
    print(accuracy)
    return output


if __name__=='__main__':
    sift = cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    images, logos = load_trainset('./data')
    logo_ids = []
    progress_bar = tqdm(total=len(images), desc="Training")
    for i in range(len(images)):
        img = images[i]
        for logo_bbox in logos[i]:
            x, y, w, h = logo_bbox.x, logo_bbox.y, logo_bbox.w, logo_bbox.h
            
            logo_img = img[y:y+h, x:x+w]
            _, descriptors = sift.detectAndCompute(logo_img, None)
            flann.add([descriptors])
            logo_ids.append(logo_bbox.logo_id)
        progress_bar.update()
    flann.train()

    test_and_evaluate(sift, flann, logo_ids)


