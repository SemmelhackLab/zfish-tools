import math

import numpy as np


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def angle_abc(a, b, c):
    ang_ab = np.arctan2(*(b - a)[::-1])
    ang_cb = np.arctan2(*(b - c)[::-1])
    return np.rad2deg((ang_cb - ang_ab) % (2 * np.pi))


def vector(a, b):
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    return dx, dy


def angle_ab(a, b):
    dx, dy = vector(a, b)
    angle = math.atan2(dy, dx)
    if angle < 0:
        angle += (2 * math.pi)
    return angle


def angle_to_vector(rad):
    theta = rad % (2 * math.pi)
    v = np.array([1, math.tan(theta)])
    v /= np.linalg.norm(v)
    if math.pi / 2 < theta <= 3 * math.pi / 2:
        v *= -1
    return v


def smaller_angle(theta1, theta2):
    diff = (theta1 - theta2) % (2 * np.pi)
    return min([diff, 2 * np.pi - diff])


def line_sort(p):
    points = p.copy()
    for i in range(1, len(points)):
        argmin = i + ((points[i:] - points[i - 1]) ** 2).sum(axis=-1).argmin()
        points[[i, argmin]] = points[[argmin, i]]
    return points


def sample_points(a, n):
    try:
        return a[np.linspace(0, len(a) - 1, n).astype(int)]
    except IndexError:
        return a
