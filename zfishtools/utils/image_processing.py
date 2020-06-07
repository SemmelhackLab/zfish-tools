import cv2
import numpy as np
from skimage.filters import rank
from skimage.morphology import disk


def crop(img, r):
    return img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]


def gaussian_blur(src, k_size):
    if k_size % 2 == 0:
        k_size += 1
    return cv2.GaussianBlur(src, (k_size, k_size), 0)


def local_otsu(src, radius, offset=0):
    if radius == 0:
        return src
    local_threshold = rank.otsu(src, disk(radius))
    return np.array(src >= local_threshold + offset, dtype='uint8')


def find_contours(thresh):
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    return sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
