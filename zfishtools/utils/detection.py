import numpy as np

from .geometry import angle_ab


def correct_orientation(left_contour, right_contour, swim_bladder_contour):
    orientation = angle_ab(swim_bladder_contour.center, (left_contour.center + right_contour.center) // 2) % (2 * np.pi)
    left_contour.correct_orientation(orientation)
    right_contour.correct_orientation(orientation)
    swim_bladder_contour.override_orientation(orientation)
    return left_contour, right_contour, swim_bladder_contour

