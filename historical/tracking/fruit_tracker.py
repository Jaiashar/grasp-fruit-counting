'''
    File name:
    Author: Chao Qu and Xu Liu
    Date created:
    Date last modified:
    Python Version: 2.7
'''

from __future__ import (print_function, division, absolute_import)

import logging
from itertools import izip

import cv2
import numpy as np

from scpye.improc.image_processing import enhance_contrast
from scpye.track.assignment import hungarian_assignment
from scpye.track.bounding_box import (bboxes_assignment_cost, extract_bbox)
from scpye.track.fruit_track import FruitTrack
from scpye.track.optical_flow import (calc_optical_flow, calc_average_flow)
from scpye.utils.drawing import (Colors, draw_bboxes, draw_optical_flows,
                                 draw_text, draw_bboxes_matches, draw_line)


# TODO: all inputs for the tracking part: images, bbox(contains a list of [x, y, w, h]),
# TODO: adjustable parameters: state_cov(P0, state covariance matrix) and proc_cov(Q, transition cov matrix representing how confident we are in our state trans matrix F, refer to kalman_filter.py)
# TODO: adjustable parameters: min_age=3(how many tracked times are regarded as valid tracks), win_size(window_size of fruits), max_level (levels of pyramid), init_flow (40,0) (initial flow velocity(automatically updated using avarage_flow));
# TODO: adjustable parameters: unassigned_cost=1.2 in assignment.py file, this controls threshold of hungarian assignment (if cost > un.._cost, this assignment will be classified to unassgnment)
# TODO: adjustable parameters: : margin pixels = COUNT_CUTOUT_PADDING, should be the same as Steven's code, control the number of margin pixels when extracting bounding boxes
# TODO: adjustable parameters: flow_cov(2,2) and bbox_cov (1,1): model noise of optical flow and bbox detection (if both are 0, no updates by KF, directly use the result of optical flow and detection)
# TODO: update:understanding of the visulization: the bboxes are final (optical flow and kalman filter) bbox positions, the lines of tracking are only optical flow traking process (not including the final update)
# TODO: understanding of kalman filter's usage: both in optical flow updates and in detection updates, and the init_flow is the same, which is mean_optical_flow, flow_cov and bbox_cov: model noise of optical flow and bbox detection
# TODO: intuition of adding KF: the general moving direction and distance should be similar for all fruits
# TODO: further understand flow_cov and bbox_cov: the row vector is diagnoalized in kalman_filter.py file, the two elements represent x and y noises (which are independent here) of the optical flow or detection model.

class FruitTracker(object):
    COUNT_CUTOUT_PADDING = 1
    # COUNT_STD_SIZE = (128,128)
    def __init__(self, min_age=3, win_size=31, max_level=3, init_flow=(-20, 0),
                 state_cov=(5, 5, 5, 5), proc_cov=(8, 4, 4, 2),
                 flow_cov=(2, 2), bbox_cov=(0, 0), margin_pixels = COUNT_CUTOUT_PADDING):
        """
        :param min_age: minimum age of a tracking to be considered for counting
        """
        self.margin_pixels = margin_pixels
        # Tracking
        self.tracks = []
        # TODO: age measures how many times the bbox is tracked
        self.min_age = min_age
        self.total_counts = 0
        self.frame_counts = 0

        self.prev_gray = None
        # Optical flow parameters
        self.win_size = win_size
        self.max_level = max_level

        # Kalman filter parameters
        self.state_cov = np.array(state_cov)
        self.init_flow = np.array(init_flow)
        self.proc_cov = np.array(proc_cov)
        self.flow_cov = np.array(flow_cov)
        self.bbox_cov = np.array(bbox_cov)

        self.logger = logging.getLogger(__name__)
        # Visualization
        self.vis = True
        self.disp_bgr = None
        self.disp_bw = None

    @property
    def initialized(self):
        return self.prev_gray is not None


    def add_new_tracks(self, tracks, fruits):
        """
        Add new fruits to tracks
        :param tracks: a list of tracks
        :param fruits: a list of fruits
        """
        for fruit in fruits:
            track = FruitTrack(fruit, self.init_flow, self.state_cov,
                               self.proc_cov)
            tracks.append(track)

    # TODO: track here contain all info about the fruit, age, bbox, pos, vel, etc. (instance of FruitTrack class which corresponds to fruit), tracks is a list contains all track
    # TODO: fruits here are a 2D array of 4 vertices of bboxes extracted in bolb_analyzer.py
    # TODO: add prev_bgr parameter to input previous frame (for counting)
    def track(self, image, fruits, bw, prev_bgr):
        """
        Main tracking step
        :param image: greyscale image --> no, it is color image
        :param fruits: new fruits #TODO: fruits here are a 2D array of [x y w h] of bboxes extracted in bolb_analyzer.py
        :param bw: binary image
        """
        # Convert to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.disp_bgr = enhance_contrast(image)
        self.disp_bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        # VISUALIZATION: new detection
        # if self.vis:
        #     draw_bboxes(self.disp_bgr, fruits, color=Colors.blue)
        #     draw_bboxes(self.disp_bw, fruits, color=Colors.blue)

        # Initialization
        if not self.initialized:
            self.prev_gray = gray
            self.add_new_tracks(self.tracks, fruits)
            self.logger.info('Tracker initialized.')
            return

        # TODO: here is just the first step of kalman filter: state prediction based on prior
        self.predict_tracks()
        self.logger.debug("Predicted tracks: {}".format(len(self.tracks)))

        # VISUALIZATION: after prediction
        # if self.vis:
        #     predict_bboxes = [t.bbox for t in self.tracks]
        #     draw_bboxes(self.disp_bgr, predict_bboxes, color=Colors.red)
        #     draw_bboxes(self.disp_bw, predict_bboxes, color=Colors.red)


        # TODO: in this function, both optical flow and kalman filter (for position correction) are used to generate two returns: updated track and lost track
        updated_tracks, lost_tracks = self.update_tracks(gray)
        self.logger.debug("update/lost: {0}/{1}".format(len(updated_tracks),
                                                        len(lost_tracks)))

        # VISUALIZATION: after optical flow update
        # if self.vis:
        #     updated_bboxes = [t.bbox for t in updated_tracks]
        #     draw_bboxes(self.disp_bgr, updated_bboxes, color=Colors.cyan)
        #     draw_bboxes(self.disp_bw, updated_bboxes, color=Colors.cyan)

        # TODO:match_tracks outputs the match detection results and predictions of optical flow + KF
        #TODO: here the matched_tracks are correted bboxes (detection and optical-flow-KF-prediction combined)
        matched_tracks, new_fruits, unmatched_tracks = self.match_tracks(
            updated_tracks,
            fruits)
        self.logger.debug("matched/new/unmatched: {0}/{1}/{2}".format(
            len(matched_tracks), len(new_fruits), len(unmatched_tracks)
        ))

        # VISUALIZATION: after hungarian assignment update
        # if self.vis:
        #     matched_bboxes = [t.bbox for t in matched_tracks]
        #     draw_bboxes(self.disp_bgr, matched_bboxes, color=Colors.green)
        #     draw_bboxes(self.disp_bw, matched_bboxes, color=Colors.green)

        # Assemble all lost tracks and update tracks
        self.tracks = matched_tracks
        self.add_new_tracks(self.tracks, new_fruits)
        lost_tracks.extend(unmatched_tracks)
        self.logger.debug(
            "tracks/lost: {0}/{1}".format(len(self.tracks), len(lost_tracks)))

        # VISUALIZATION:

        if self.vis:
            # diff between counted and tracked boxes: >min_age(3) or not
            counted_bboxes = [t.bbox for t in self.tracks if
                              t.age >= self.min_age]
            tracked_bboxes = [t.bbox for t in self.tracks if
                              1 < t.age <= self.min_age]
            new_bboxes = [t.bbox for t in self.tracks if t.age == 1]
            if len(counted_bboxes):
                draw_bboxes(self.disp_bgr, counted_bboxes, color=Colors.green,
                            thickness=2)
                draw_bboxes(self.disp_bw, counted_bboxes, color=Colors.green,
                            thickness=2)
            if len(tracked_bboxes):
                draw_bboxes(self.disp_bgr, tracked_bboxes, color=Colors.yellow,
                            thickness=2)
                draw_bboxes(self.disp_bw, tracked_bboxes, color=Colors.yellow,
                            thickness=2)
            if len(new_bboxes):
                draw_bboxes(self.disp_bgr, new_bboxes, color=Colors.red,
                            thickness=2)
                draw_bboxes(self.disp_bw, new_bboxes, color=Colors.red,
                            thickness=2)


        # TODO: here is visualizing the tracking in multiple concurrent frames
        if self.vis:
            for track in self.tracks:
                draw_line(self.disp_bgr, track.hist, color=Colors.yellow, thickness = 0)
                draw_line(self.disp_bw, track.hist, color=Colors.yellow, thickness = 0)


        # TODO: count the lost tracks (is valid track in previous frame but not shown in the following frames) which used to be counted tracks (min_age>=3) instead of current counted tracks!!!!!!!
        # Count fruits in lost tracks
        # TODO: add prev_bgr parameter to input previous frame(for counting)
        self.count_in_tracks(lost_tracks, prev_bgr)
        self.logger.info(
            "Frame/Total counts: {0}/{1}".format(self.frame_counts,
                                                 self.total_counts))


        if self.vis:
            h, w = np.shape(bw)
            draw_text(self.disp_bgr, self.total_counts, (10, h - 10), scale=1.5,
                      color=Colors.cyan)
            draw_text(self.disp_bw, self.total_counts, (10, h - 10), scale=1.5,
                      color=Colors.cyan)


    # TODO: here is just the first step of kalman filter: state prediction based on prior
    def predict_tracks(self):
        """
        Predict tracks in Kalman filter
        """
        for track in self.tracks:
            track.predict()


    #TODO: in this function, both optical flow and kalman filter (for position correction) are used to generate two returns: updated track and lost track
    def update_tracks(self, gray):
        """
        Update tracks' position in Kalman filter via KLT
        :param gray: greyscale image
        :return: updated tracks, lost_tracks
        """
        # Get points in previous image and points in current image
        # TODO: tracks are lists containing all info such as bbox, age, pos, etc.
        # TODO: prev_pts and init_pts here are all pixels in the fruit contour
        prev_pts = [t.prev_pos for t in self.tracks]
        init_pts = [t.pos for t in self.tracks]

        # TODO: here the status is output status vector (of unsigned chars): each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, 0.
        # TODO: can change the window_size, max_level (levels of pyramid) to change the performace
        curr_pts, status = calc_optical_flow(self.prev_gray, gray,
                                             prev_pts, init_pts,
                                             self.win_size,
                                             self.max_level)

        # Update init flow
        self.init_flow = calc_average_flow(prev_pts, curr_pts, status)
        self.logger.debug("init flow: {}".format(self.init_flow))

        # VISUALIZATION: optical flow
        # TODO: draw the optical flow tracking lines
        # draw_optical_flows(self.disp_bgr, prev_pts, curr_pts, status,
        #                    radius=2, color=Colors.magenta)
        # draw_optical_flows(self.disp_bw, prev_pts, curr_pts, status,
        #                    radius=2, color=Colors.magenta)


        # TODO: add tracked fruits to tracks (after updating using kalman filter), remove lost ones.
        # Remove lost tracks
        updated_tracks, lost_tracks = [], []
        for track, point, stat in izip(self.tracks, curr_pts, status):
            if stat:
                #TODO: here use kalman filter to update the position!!!
                track.correct_flow(point, self.flow_cov)
                updated_tracks.append(track)
            else:
                lost_tracks.append(track)

        self.prev_gray = gray

        return updated_tracks, lost_tracks


    # TODO: use the hungarian_assignment to match_tracks outputs the match detection results and predictions of optical flow + KF
    # TODO: fruits here are a 2D array of 4 vertices of bboxes extracted in bolb_analyzer.py
    # TODO: track here is the instance of FruitTrack class which corresponds to fruit, tracks is a list contains all track




    def match_tracks(self, tracks, fruits):
        """
        Match tracks to new detection
        :param tracks:
        :param fruits:#TODO: fruits here are a 2D array of [x y w h] of bboxes extracted in bolb_analyzer.py
        :return: matched_tracks, new_fruits, unmatched_tracks
        """
        # TODO: t.box should be 2d array; fruits should be 3d array
        bboxes_update = np.array([t.bbox for t in tracks])
        bboxes_detect = np.array(fruits)

        # TODO: this cost contains two parts: overlap ratio cost is 1-overlap ratio; distsq_area_ratio_cost is bbox_distsq_area_ratio computed in the function.
        # TODO: Intuition of this cost: the larger overlap ratio and smaller distance/area ratio will result in smaller cost
        cost = bboxes_assignment_cost(bboxes_update, bboxes_detect)

        # TODO: refer to the assignment.py file
        match_inds, lost_inds, new_inds = hungarian_assignment(cost)

        # VISUALIZATION: hungarian assignment
        # draw_bboxes_matches(self.disp_bgr, match_inds, bboxes_update,
        #                     bboxes_detect, color=Colors.cyan)
        draw_bboxes(self.disp_bgr, bboxes_detect, color=Colors.cyan,
                    thickness=1)
        draw_bboxes(self.disp_bw, bboxes_detect, color=Colors.cyan,
                    thickness=1)




        # get matched tracks
        matched_tracks = []
        #TODO: here the bboxes are correted from the original detections to detection-prediction combined result
        for match in match_inds:
            i_track, i_fruit = match
            track = tracks[i_track]
            track.correct_bbox(fruits[i_fruit], self.bbox_cov)
            matched_tracks.append(track)



        # extract new tracks
        new_fruits = fruits[new_inds]

        # get unmatched tracks
        unmatched_tracks = [tracks[ind] for ind in lost_inds]

        return matched_tracks, new_fruits, unmatched_tracks


    # TODO: count the lost tracks which used to be counted tracks (min_age>=3) instead of current counted tracks!!!!!!!
    # TODO: add prev_bgr parameter to input previous frame(for counting)
    def count_in_tracks(self, tracks, prev_bgr):
        """
        Count how many fruits there are in tracks
        :param tracks: list of tracks
        """
        #TODO: if t.age>3, take that as a valid count, directly add all bboxes together, instead of counting how many fruits are in every bboxes
               #TODO: NOT AS EASY AS THOUGHT!!! need to extract PREVIOUS position (prov_pos) of bboxes and extract bbox images in PREVIOUS frame!!!
        #TODO: use the frame_bboxes (list of bboxes (x, y, w, h)) to extract bboxes and count use our FCN!!!
        frame_bboxes = []
        frame_bbox_imgs = []
        for t in tracks:
            if t.age >= self.min_age:
                frame_bboxes.append(t.bbox)
                # y1 = t.bbox[1]
                # y2 = t.bbox[1]+t.bbox[3]
                # x1 = t.bbox[0]
                # x2 = t.bbox[0]+t.bbox[2]
                # bbox_img_temp = self.disp_bgr[y1:y2,x1:x2,:]
                # TODO: get the bbox position in previous frame!(for counting)
                bbox_pre_fr = t.bbox
                bbox_pre_fr[0] += t.prev_pos[0] - t.pos[0]
                bbox_pre_fr[1] += t.prev_pos[1] - t.pos[1]
                #TODO: add margin to bounding boxes so that the bboxes are more complete
                bbox_pre_fr[0] -= self.margin_pixels
                bbox_pre_fr[1] -= self.margin_pixels
                bbox_pre_fr[2] += 2*self.margin_pixels
                bbox_pre_fr[3] += 2*self.margin_pixels

                # print ('pos0,pos2',bbox_pre_fr[0],bbox_pre_fr[2],'\n')
                bbox_img_temp = extract_bbox(prev_bgr, bbox_pre_fr, copy=False)
                h,w,_ = bbox_img_temp.shape
                # TODO: can remove this, no out-of-boundary index
                if not (h*w):
                    # print('h=%d w=%d' %(h,w), '\n')
                    continue
                # TODO:resize to the size COUNT_STD_SIZE (in Steven's constants.py)
                bbox_img_temp = cv2.resize(bbox_img_temp, (128,128))
                frame_bbox_imgs.append(bbox_img_temp)
                # print ('bbox_img_shape', bbox_img_temp.shape, '\n')

                # TODO: visualize the source of the lost tracks (our counted bboxes) in current frame
                # cv2.imshow('bbox_img_temp',bbox_img_temp)
                # cv2.waitKey(0)
                # draw_bboxes(prev_bgr, bbox_pre_fr, color=Colors.green,
                #             thickness=3)
                # print('***********************************************')
        # TODO: VERY IMPORTANT!!!!!!!!! use our counting network to count!!!
        # TODO: frame_bbox_imgs includes all bbox images for this frame, maybe here needs a loop to count!!!
        # TODO: my plan: use Steven's code to read
        # def get_bbox_count(frame_bbox_imgs)
            #run FCN prediction here
        # self.frame_counts = get_bbox_count(frame_bbox_imgs)

        self.frame_counts = sum([1 for t in tracks if t.age >= self.min_age])
        # print('frame_counts:', self.frame_counts, '\n\n')
        self.total_counts += self.frame_counts

    def finish(self, prev_bgr):
        self.count_in_tracks(self.tracks, prev_bgr)
        self.logger.info("Total counts: {}".format(self.total_counts))
        print('total counts are:', self.total_counts, '\n')