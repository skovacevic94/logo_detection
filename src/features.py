from tqdm import tqdm
import numpy as np
import cv2
from scipy.cluster.vq import kmeans, vq
from skimage.feature import hog


def create_vocabulary(images, k=130):
    sift = cv2.SIFT_create()
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


def compute_sift_bow_features(images, voc):
    sift = cv2.SIFT_create()
    features = np.zeros((len(images), len(voc)))
    for i, img in enumerate(tqdm(iterable=images, desc="Computing features (Quantization + Histogram)")):
        _, desc = sift.detectAndCompute(img, None)
        if desc is None:
            continue
        words,distance = vq(desc, voc)
        for w in words:
            features[i,w] += 1
    return features


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation = inter)


def prepare_size(image, dim, keep_ratio=True, inter = cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if not keep_ratio or w == h:
        return cv2.resize(image, (dim, dim), interpolation=inter)

    if w > h:
        resized = image_resize(image, width=dim, inter=inter)
    else:
        resized = image_resize(image, height=dim, inter=inter)
    res = np.zeros((dim, dim), dtype=np.uint8)
    res[0:resized.shape[0], 0:resized.shape[1]] = resized
    return res


def compute_hog_features(images, scale_factor=None):
    features = []
    dim = 64
    for i, img in enumerate(images):
        img = prepare_size(img, dim, keep_ratio=False, inter=cv2.INTER_LINEAR)
        H = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        features.append(H)
    return np.vstack(features)