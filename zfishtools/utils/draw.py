import cv2

from .geometry import  angle_to_vector


def draw_contour(img, contour, color=(0, 0, 0), thickness=-1):
    cv2.drawContours(img, [contour], -1, color, thickness=thickness)


def draw_point(img, point, radius=3, color=(0, 0, 0)):
    cv2.circle(img, (int(round(point[0])), int(round(point[1]))), radius=radius, color=color, thickness=-1)


def draw_line(img, point, angle, length, color, two_sided=False):
    dx, dy = angle_to_vector(angle) * length
    if two_sided:
        p1 = (int(round(point[0] - dx)), int(round(point[1] - dy)))
    else:
        p1 = (int(round(point[0])), int(round(point[1])))
    p2 = (int(round(point[0] + dx)), int(round(point[1] + dy)))
    cv2.line(img, p1, p2, color=color, lineType=4)
