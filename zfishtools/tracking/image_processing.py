import cv2

from .analysis import get_body_part_contours, get_tail_points
from zfishtools.utils import draw_line, draw_contour, draw_point


def draw_body_parts(src, display, rois, radius, threshold, tail_points=(), **kwargs):
    left_eye, right_eye, swim_bladder = get_body_part_contours(src, rois, radius, threshold)

    img = src.copy()
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]

    for cnt, color in zip([left_eye, right_eye, swim_bladder], colors):
        draw_contour(img, cnt.convex_hull, thickness=-1)
        for p in cnt.points:
            draw_point(img, p, color=color)

        draw_line(img, cnt.center, cnt.orientation, 200, color=color)

    for p in tail_points:
        draw_point(img, p, color=(255, 255, 0))

    display.image_kwargs['eye_points'] = left_eye.points, right_eye.points, swim_bladder.points

    return img
