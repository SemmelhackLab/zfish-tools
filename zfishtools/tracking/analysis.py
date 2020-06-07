import cv2
import numpy as np
from skimage.morphology import skeletonize

from zfishtools.utils import crop, local_otsu, BodyPartContour, correct_orientation, line_sort, sample_points


def get_body_part_contours(src, rois, radius, threshold):
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    body_part_crops = [crop(img, roi) for roi in rois]
    thresholds = [local_otsu(img, radius, threshold) for img in body_part_crops]
    contours = [BodyPartContour(th, roi, img.shape) for th, roi in zip(thresholds, rois)]
    return correct_orientation(*contours)


def get_tail_points(src, color, n_points):
    c = np.array(color)
    mask = np.all(src == c, axis=2)
    skeleton = skeletonize(mask)
    points = np.argwhere(skeleton)
    points_sorted = line_sort(points)
    points_selected = sample_points(points_sorted, n_points)
    return points_selected[:, ::-1]
