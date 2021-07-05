from tqdm import tqdm
import numpy as np
import cv2
from skimage.feature import hog


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


def compute_hog_features(images):
    features = None
    for i, img in enumerate(tqdm(iterable=images, desc="Computing features (Quantization + Histogram)")):
        dim = 64
        img = prepare_size(img, dim, keep_ratio=False, inter=cv2.INTER_LINEAR)
        H = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        if features is None:
            features = H
        else:
            features = np.vstack((features, H))
    return features