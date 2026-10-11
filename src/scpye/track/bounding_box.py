'''
    File name:
    Author: Chao Qu
    Date created:
    Date last modified:
    Python Version: 2.7
'''
from __future__ import (print_function, division, absolute_import)

import numpy as np


class OverlapRatio:
    """
    Min: intersection area / min(bbox1 area, bbox2 area)
    Union: intersection area / union(bbox1 area, bbox2 area)
    """
    Union, Min = range(2)

    def __init__(self):
        pass


#TODO: here the input is image and bbox, bbox is a list which contains x, y, w, h
def extract_bbox(image, bbox, copy=False):
    """
    Extract bbox from image
    :param image: image
    :param bbox: bbox
    :param copy: whether to return a copy or reference
    :return: region of image
    """
    x, y, w, h = bbox
    if copy:
        return np.array(image[y:y + h, x:x + w, ...], copy=True)
    else:
        return image[y:y + h, x:x + w, ...]


#TODO: here the input is new center???? and bbox, bbox is a list which contains x, y, w, h
def shift_bbox(bbox, center):
    """
    Shift bbox to new center
    :param bbox:
    :param center:
    :return:
    """
    x, y, w, h = bbox
    xn, yn = center
    return np.array([xn - w / 2, yn - h / 2, w, h], dtype=int)


#TODO: here the input is bbox, calculate its area
def bbox_area(bbox):
    """
    Area of bbox
    :param bbox: bbox
    :return: bbox area
    """
    return bbox[-1] * bbox[-2]

#TODO: here the input is bbox, calculate its canter
def bbox_center(bbox):
    """
    Center of a bounding box
    :param bbox: bbox
    :return: center of bbox
    """
    x, y, w, h = bbox
    return np.array([x + w / 2.0, y + h / 2.0])

#TODO: here input are two bounding boxes, calculate the distance between them, used the center function
def bbox_distsq(bbox1, bbox2):
    """
    Squared distance between bbox1 and bbox2
    :param bbox1: bbox1
    :param bbox2: bbox2
    :return: squared distance between center of bbox
    """
    cx1, cy1 = bbox_center(bbox1)
    cx2, cy2 = bbox_center(bbox2)
    return (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

#TODO: here check whether two bbox have intersection, if yes, return true
def bbox_intersect(bbox1, bbox2):
    """
    Whether two bboxes intersect or not
    :param bbox1: bbox1
    :param bbox2: bbox2
    :return: True if two bboxes intersect
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    dx = abs((x1 + w1 / 2) - (x2 + w2 / 2))
    dy = abs((y1 + h1 / 2) - (y2 + h2 / 2))
    return (dx * 2 < (w1 + w2)) and (dy * 2 < (h1 + h2))

#TODO: calculate the intersection area of two bbox
def bbox_intersect_area(bbox1, bbox2):
    """
    Intersection area of two bboxes
    :param bbox1: bbox1
    :param bbox2: bbox2
    :return: intersection area
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    w = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    h = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return w * h

#TODO: calculate the overlap ratio of two bbox, used the intersect_area and bbox_area functions, the ratio is inter_area/(min area of bbox)
def bbox_overlap_ratio(bbox1, bbox2, ratio_type=OverlapRatio.Union):
    """
    Overlap ratio of two bboxes
    :param bbox1: bbox
    :param bbox2: bbox
    :param ratio_type:
    :return: overlap ratio
    """
    if not bbox_intersect(bbox1, bbox2):
        return 0.0

    # intersection area is a * b, where a is width and b is height
    intersect_area = bbox_intersect_area(bbox1, bbox2)
    bbox1_area = bbox_area(bbox1)
    bbox2_area = bbox_area(bbox2)

    if ratio_type == OverlapRatio.Union:
        area_union = bbox1_area + bbox2_area - intersect_area
        return intersect_area / area_union
    elif ratio_type == OverlapRatio.Min:
        area_min = min(bbox1_area, bbox2_area)
        return intersect_area / area_min

#TODO: calculate dis/2*(sum of bbox area) use the bbox_distsq function
def bbox_distsq_area_ratio(bbox1, bbox2):
    """

    :param bbox1:
    :param bbox2:
    :return:
    """
    dist_sq = bbox_distsq(bbox1, bbox2)
    area = bbox_area(bbox1) + bbox_area(bbox2)
    return dist_sq / area / 2.0

#TODO: input two sets of bboxes, return their pair bbox overlap ratio (every bbox in set 1 are calculated wrt. every one in set 2)
#TODO: return a 2D numpy array R,  in which R[i,j] is ratio of ith set 1 bbox and jth set 2 bbox
def bboxes_overlap_ratio(bboxes1, bboxes2, ratio_type=OverlapRatio.Union):
    """

    :param bboxes1:
    :param bboxes2:
    :param ratio_type:
    :return:
    """
    bboxes1 = np.atleast_2d(bboxes1)
    bboxes2 = np.atleast_2d(bboxes2)
    n1 = len(bboxes1)
    n2 = len(bboxes2)
    assert n1 > 0 and n2 > 0

    R = np.zeros((n1, n2))
    for i1, b1 in enumerate(bboxes1):
        for i2, b2 in enumerate(bboxes2):
            R[i1, i2] = bbox_overlap_ratio(b1, b2, ratio_type)
    return R

def bboxes_assignment_cost(bboxes1, bboxes2):
    """

    :param bboxes1:
    :param bboxes2:
    :return:
    """
    bboxes1 = np.atleast_2d(bboxes1)
    bboxes2 = np.atleast_2d(bboxes2)

    n1 = len(bboxes1)
    n2 = len(bboxes2)
    assert n1 > 0 and n2 > 0

    C = np.zeros((n1, n2))
    for i1, b1 in enumerate(bboxes1):
        for i2, b2 in enumerate(bboxes2):
            overlap_ratio = bbox_overlap_ratio(b1, b2)
            overlap_ratio_cost = 1 - overlap_ratio
            distsq_area_ratio_cost = bbox_distsq_area_ratio(b1, b2)
            C[i1, i2] = overlap_ratio_cost + distsq_area_ratio_cost
    return C
