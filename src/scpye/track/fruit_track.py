'''
    File name:
    Author: Chao Qu
    Date created:
    Date last modified:
    Python Version: 2.7
'''

from __future__ import (print_function, division, absolute_import)

import numpy as np
from scpye.track.kalman_filter import KalmanFilter
from scpye.track.bounding_box import (bbox_center, shift_bbox)
'''
    File name:
    Author: Chao Qu
    Date created:
    Date last modified:
    Python Version: 2.7
'''

def cov2ellipse(P, ns=3):
    """
    Covariance to ellipse
    :param P: covariance
    :param ns: scale
    :return:
    """
    assert np.shape(P) == (2, 2)

    U, D, _ = np.linalg.svd(P)
    w, h = np.sqrt(D) * ns
    a = np.arctan2(U[1, 0], U[0, 0])

    return w, h, np.rad2deg(a)


class FruitTrack(object):
    def __init__(self, bbox, init_flow, state_cov, proc_cov):
        self.bbox = bbox
        self.age = 1
        self.hist = []
        self.prev_pos = None

        # TODO:every track will initialize its own kalman filter
        # every track will initialize its own kalman filter
        init_pos = bbox_center(self.bbox)
        init_state = np.hstack((init_pos, init_flow))

        self.kf = KalmanFilter(x0=init_state, P0=state_cov, Q=proc_cov)
        self.hist.append(self.pos)

    @property
    def pos(self):
        return self.kf.x[:2].copy()

    @property
    def vel(self):
        return self.kf.x[2:].copy()

    @property
    def pos_cov(self):
        return self.kf.P[:2, :2].copy()

    @property
    def cov_ellipse(self):
        wha = cov2ellipse(self.pos_cov)
        return np.hstack((self.pos, wha))

    def predict(self):
        """
        Predict new location of the tracking
        """
        self.prev_pos = self.pos
        self.kf.predict()
        self.bbox = shift_bbox(self.bbox, self.pos)

    # TODO: here the bboxes are correted using optical flow and
    def correct_flow(self, pos, flow_cov):
        """
        Correct location of the track from pos input
        :param pos:
        :param flow_cov:
        """
        self.kf.update_pos(pos, flow_cov)
        self.bbox = shift_bbox(self.bbox, self.pos)

    # TODO: here the bboxes are correted from the original detections to detection-prediction combined result
    def correct_bbox(self, bbox, bbox_cov):
        """
        Correct location of the track hungarian assignment
        :param bbox:
        :param bbox_cov:
        :return:
        """
        pos = bbox_center(bbox)
        self.kf.update_pos(pos, bbox_cov)
        # Here we shift using current bbox (fixed)
        self.bbox = shift_bbox(bbox, self.pos)

        # Increment age and add track to hist
        # TODO: age measures how many times the bbox is tracked
        self.age += 1
        self.hist.append(self.pos)
