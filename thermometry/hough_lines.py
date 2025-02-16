# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import cv2
import xarray as xr
import numpy as np

from skimage import draw
from skspatial.objects import Line

class HoughLine():

    def __init__(self, line, Z=None):
        x1, y1, x2, y2 = line
        self.pt1 = np.array([x1, y1])
        self.pt2 = np.array([x2, y2])
        self.xs = np.array([x1, x2])
        self.ys = np.array([y1, y2])
        self.angle = np.rad2deg(np.arctan2(np.diff(self.ys), np.diff(self.xs)))

        if Z is not None:
            self.calculate_score_from(Z)

    def draw_line(self) -> tuple[np.ndarray, np.ndarray]:
        # draw discretized line on a pixel grid
        return draw.line(*self.pt1, *self.pt2)

    def calculate_score_from(self, Z):
        rr, cc = self.draw_line()
        pts = Z[rr,cc]

        self.score = np.sum(pts)
        self.std = np.std(pts)

        return self.score

    def get_line(self):
        return Line.from_points(self.pt1, self.pt2)

    def get_metric(self, metric, Z, buffer=0):
        Y, X = self.draw_line()
        score = 0
        Ymin, Ymax = Y[0], Y[-1]
        if Ymin > Ymax:
            Ymin, Ymax = Ymax, Ymin
        for i in range(-buffer, buffer+1):
            if Ymin+i > 0 and Ymax+i < Z.shape[1]:
                score += metric(Z[X, Y+i])
        return score

    def get_L1norm(self, Z, buffer=0):
        den = self.get_metric(np.sum, np.ones_like(Z), buffer)
        if den==0:
            return 0
        return self.get_metric(np.sum, Z, buffer) / den

    def get_sum(self, Z, buffer=0):
        return self.get_metric(np.sum, Z, buffer)

    def get_len(self, Z, buffer=0):
        return self.get_metric(np.sum, np.ones_like(Z), buffer)

    def get_std(self, Z, buffer=0):
        return self.get_metric(np.std, Z, buffer)

    def save_params(self, Z, buffer=0):
        self.std = self.get_std(Z, buffer)
        self.L1norm = self.get_L1norm(Z, buffer)
        self.len = self.get_sum(Z, buffer)
        self.sum = self.get_len(Z, buffer)

    def dist(self, houghline):
        return self.get_line().distance_line(houghline.get_line())

    def unique_by_dist(self, houghline, dist_cutoff):
        if houghline is None:
            return True
        return self.dist(houghline) > dist_cutoff

    def distance_to_point(self, point):
        return np.linalg.norm(np.cross(self.pt2-self.pt1, self.pt1-np.array(point)))/np.linalg.norm(self.pt2-self.pt1)

    def get_midpoint(self, preserve_trpe=True):
        midpoint = (self.pt1 + self.pt2)/2
        if not preserve_trpe:
            return midpoint
        return midpoint.astype(self.pt1.dtype)

def normalize_image(
        Z: np.ndarray,
        blur: tuple | int = 1,
        erode_iter: tuple | int = 10,
        erode_size: tuple | int = 30,
        dilate_size: tuple | int = (1,1),
        debug: bool = False,
        debug_aspect: tuple | None = None,
        resize_vert: int = 1
        ) -> np.ndarray:
    # rescale input and use morphology to convert gate-gate map into a HoughLines ready format
    Z = Z.copy()
    Z -= np.min(Z)
    Z *= 256/np.max(Z)
    Z = Z.astype(np.uint8)

    if resize_vert != 1:
        Z = cv2.resize(Z, (Z.shape[1], int(round(resize_vert*Z.shape[0]))),
                       interpolation = cv2.INTER_LINEAR)

    if isinstance(dilate_size, int):
        dilate_kernel = np.ones((dilate_size, dilate_size))
    else:
        dilate_kernel = np.ones(dilate_size)

    if isinstance(erode_size, int):
        erode_size = (erode_size, erode_size)

    Z = cv2.medianBlur(Z, blur)
    Z = cv2.dilate(Z, dilate_kernel)
    Z = cv2.erode(Z, erode_size, iterations=erode_iter)

    Z = np.where(Z>np.median(Z), Z, 0)
    if Z[Z>0].shape != (0, ):
        Z[Z>0] -= Z[Z>0].min()

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(Z, origin="lower", aspect=debug_aspect)
        plt.show()

    return Z

def find_houghlines(
        input_Z: np.ndarray,
        resize_vert: int = 100,
        norm_kwargs: dict = {},
        hough_kwargs: dict = dict(minLineLength=30, maxLineGap=200)
    ) -> list[HoughLine]:
    # Find Lines Using a Probabilistic Hough lines method

    # exponentiate to sharpen lines
    sharpened_Z = input_Z**6

    norm_kwargs["resize_vert"] = resize_vert
    Z = normalize_image(sharpened_Z, **norm_kwargs)

    lines = cv2.HoughLinesP(Z, 1, np.pi/180, 100, **hough_kwargs)
    if lines is None:
        return []

    sol_lines = []
    for line in lines:

        line = np.array([[i[0], int(np.floor(i[1]/resize_vert)), i[2], int(np.floor(i[3]/resize_vert))] for i in line])
        newline = HoughLine(line[0])

        skip = False
        for pline in sol_lines:
            d = pline.distance_to_point(newline.get_midpoint())
            if d<7:
                skip = True
                continue
        if not skip:
            newline.save_params(sharpened_Z, 10)
            sol_lines.append(newline)

    return sol_lines
