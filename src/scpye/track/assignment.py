# Modified for the October 2026 portfolio release; see PROVENANCE.md.
'''
    File name:
    Author: Chao Qu
    Date created:
    Date last modified:
    Python Version: 2.7
'''

import numpy as np
from scipy.optimize import linear_sum_assignment


def pad_cost_matrix(cost_matrix, unassigned_cost):
    """
    Pad cost matrix to handle unassignment
    :param cost_matrix: M x N matrix, M - tracks, N - detections
    :param unassigned_cost:  cost of unassignment
    :return: padded cost matrix
    """

    # Leaving both endpoints unmatched costs twice unassigned_cost.
    if unassigned_cost == 0.0:
        return cost_matrix

    r, c = np.shape(cost_matrix)
    # padded size
    n = r + c
    float_max = np.inf
    padded_cost_matrix = np.ones((n, n)) * float_max

    # fill the padded cost matrix
    padded_cost_matrix[:r, :c] = cost_matrix
    padded_cost_matrix[r:, c:] = 0.0
    padded_cost_matrix[range(r), range(c, n)] = unassigned_cost
    padded_cost_matrix[range(r, n), range(c)] = unassigned_cost
    return padded_cost_matrix


def hungarian_assignment(cost_matrix, unassigned_cost=1.2):
    """
    This is the equivalent of matlab's assignDetectionsToTracks
    :param cost_matrix: M x N matrix, M - tracks, N - detections
    :param unassigned_cost: cost of unassignment
    :return: matches, unassigned tracks, unassigned detections
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if cost_matrix.ndim != 2 or not np.all(np.isfinite(cost_matrix)):
        raise ValueError("cost_matrix must be a finite two-dimensional matrix")
    if not np.isfinite(unassigned_cost) or unassigned_cost <= 0:
        raise ValueError("unassigned_cost must be finite and positive")
    if 0 in cost_matrix.shape:
        return (np.empty((0, 2), dtype=int), np.arange(cost_matrix.shape[0]),
                np.arange(cost_matrix.shape[1]))
    padded_cost_matrix = pad_cost_matrix(cost_matrix, unassigned_cost)
    rows, columns = linear_sum_assignment(padded_cost_matrix)
    assignment = np.column_stack((rows, columns))

    # Figure out matches, unassigned bboxes1 and unassigned bboxes2
    n1, n2 = np.shape(cost_matrix)
    ind1 = assignment[:, 0]
    ind2 = assignment[:, 1]

    # Correct matches
    match_inds = (ind1 < n1) & (ind2 < n2)
    matches = assignment[match_inds]
    matches = np.atleast_2d(matches)

    # Unassigned bboxes1 and bboxes2
    un_ind1 = (ind1 < n1) & (ind2 >= n2)
    un_ind2 = (ind1 >= n1) & (ind2 < n2)

    unassigned1 = ind1[un_ind1]
    unassigned2 = ind2[un_ind2]

    return matches, unassigned1, unassigned2
