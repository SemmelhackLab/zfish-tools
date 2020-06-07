import cv2
import numpy as np

from .geometry import smaller_angle
from .draw import draw_point, draw_line, draw_contour


class BodyPartContour:
    def __init__(self, cropped_thresh, roi, img_shape):
        self.contour = max(cv2.findContours(cropped_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1],
                           key=lambda x: cv2.contourArea(x)) + roi[:2]
        self.orientation_estimate = None
        self.orientation_override = None
        self.img_shape = img_shape

    def correct_orientation(self, estimate):
        self.orientation_estimate = estimate

    def override_orientation(self, orientation):
        self.orientation_override = orientation

    @property
    def area(self):
        return cv2.contourArea(self.contour)

    @property
    def center(self):
        m = self.moments
        return np.rint(np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])).astype(int)

    @property
    def convex_hull(self):
        return cv2.convexHull(self.contour)

    @property
    def moments(self):
        return cv2.moments(self.convex_hull)

    @property
    def orientation(self):
        if self.orientation_override is not None:
            return self.orientation_override
        m = self.moments
        x = m['m10'] / m['m00']
        y = m['m01'] / m['m00']
        a = m['m20'] / m['m00'] - x * x
        b = m['m02'] / m['m00'] - y * y
        c = 2 * (m['m11'] / m['m00'] - x * y)
        angle = .5 * np.arctan(c / (a - b)) + (a < b) * np.pi / 2
        if self.orientation_estimate is not None:
            opp_angle = (angle + np.pi) % (2 * np.pi)
            if smaller_angle(self.orientation_estimate, opp_angle) < smaller_angle(self.orientation_estimate, angle):
                return opp_angle
        return angle

    @property
    def perimeter(self):
        return cv2.arcLength(self.contour, True)

    @property
    def points(self):
        zeros1 = np.zeros(self.img_shape, dtype='uint8')
        zeros2 = np.zeros(self.img_shape, dtype='uint8')
        zeros3 = np.zeros(self.img_shape, dtype='uint8')

        draw_contour(zeros1, self.convex_hull, color=1, thickness=1)
        draw_line(zeros2, self.center, self.orientation, self.perimeter, 1)
        draw_line(zeros3, self.center, self.orientation + np.pi, self.perimeter, 1)
        anterior = np.argwhere(zeros1 & zeros2)[0][::-1]
        posterior = np.argwhere(zeros1 & zeros3)[0][::-1]
        return anterior, posterior, self.center

    def draw(self, src, color=(255, 255, 255)):
        try:
            img = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        except cv2.error:
            img = src.copy()

        draw_contour(img, self.convex_hull, color=color, thickness=1)
        draw_point(img, self.center, color=color)
        draw_line(img, self.center, self.orientation, self.perimeter, color=color)
        return img
