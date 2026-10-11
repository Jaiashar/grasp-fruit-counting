# Modified for the October 2026 portfolio release; see PROVENANCE.md.
'''
    File name:
    Author: Chao Qu
    Date created:
    Date last modified:
    Python Version: 2.7
'''
#TODO: calculate the optical flow of two grayscale images

from __future__ import (print_function, division, absolute_import)

import cv2
import numpy as np


def points_inside_image(points, image, b=0):
    """
    Check if point is inside image with a certain margin
    :param points:
    :param image:
    :param b: distance from border
    :return:
    """
    h, w = np.shape(image)
    px = points[:, :, 0]
    py = points[:, :, 1]
    return (px >= b) & (px < w - b) & (py >= b) & (py < h - b)


def _make_cv_points(points):
    """
    Prepare points for opencv, convert dtype to np.float32 and add extra
    dimension
    :param points:
    :return:
    """
    points = np.array(points)
    if points.dtype is not np.float32:
        points = np.array(points, dtype=np.float32)
    if np.ndim(points) == 2:
        points = np.expand_dims(points, 1)
    return points


#TODO: points2 are optical flow results of points1
#TODO: for more, refer to doc: https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html
def calc_optical_flow(gray1, gray2, points1, points2, win_size, max_level):
    """
    Thin wrapper around opencv's calcOpticalFlowPyrLK
    :param gray1: previous image
    :param gray2: current image
    :param points1: points in previous image
    :param points2: points in current image
    :param win_size: window size
    :param max_level: max pyramid level
    :return: (tracked points shaped N x 2, boolean status shaped N)
    """
    if len(points1) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=bool)
    points1 = _make_cv_points(points1)
    points2 = _make_cv_points(points2)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    klt_params = dict(winSize=(win_size, win_size),
                      maxLevel=max_level,
                      flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                      criteria=criteria)

    # TODO: reference is here: https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
    points2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1,
                                                  points2, **klt_params)

    # is_inside = points_inside_image(points2, gray2)

    # status = (status == 1) & is_inside

    if points2 is None or status is None:
        return points1.reshape(-1, 2), np.zeros(len(points1), dtype=bool)
    return points2.reshape(-1, 2), status.reshape(-1).astype(bool)


# TODO: calculate the mean of all valid flows
def calc_average_flow(points1, points2, status):
    """
    Calculate average optical flow
    :param points1:
    :param points2:
    :param status:
    :return:
    """
    points1 = np.asarray(points1, dtype=float).reshape(-1, 2)
    points2 = np.asarray(points2, dtype=float).reshape(-1, 2)
    status = np.asarray(status, dtype=bool).reshape(-1)
    if not status.any():
        return np.zeros(2)
    return np.mean((points2 - points1)[status], axis=0)
