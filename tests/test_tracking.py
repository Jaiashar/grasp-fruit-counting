import unittest
import cv2
import numpy as np
from examples.synthetic_demo import DemoCounter, sequence
from scpye.track.assignment import hungarian_assignment
from scpye.track.kalman_filter import KalmanFilter
from scpye.track.optical_flow import calc_optical_flow, calc_average_flow


class AssignmentTests(unittest.TestCase):
    def test_rectangular_and_reordered(self):
        matches, old, new = hungarian_assignment([[8, .1, 8], [.1, 8, 8]], .75)
        self.assertEqual(matches.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(old.tolist(), [])
        self.assertEqual(new.tolist(), [2])

    def test_unassignment_pays_for_both_endpoints(self):
        self.assertEqual(len(hungarian_assignment([[1.1]], .75)[0]), 1)
        matches, old, new = hungarian_assignment([[1.6]], .75)
        self.assertEqual(matches.shape, (0, 2))
        self.assertEqual(old.tolist(), [0])
        self.assertEqual(new.tolist(), [0])

    def test_empty_shapes(self):
        for n, m in [(0, 0), (0, 3), (2, 0)]:
            matches, old, new = hungarian_assignment(np.empty((n, m)))
            self.assertEqual(matches.shape, (0, 2))
            self.assertEqual(old.tolist(), list(range(n)))
            self.assertEqual(new.tolist(), list(range(m)))

    def test_invalid_input(self):
        for matrix, cost in [([1], 1), ([[np.nan]], 1), ([[1]], 0)]:
            with self.assertRaises(ValueError):
                hungarian_assignment(matrix, cost)


class MotionTests(unittest.TestCase):
    def test_kalman_predict_and_correct(self):
        kf = KalmanFilter(x0=np.array([10., 20., 2., -1.]))
        kf.predict()
        np.testing.assert_allclose(kf.x, [12, 19, 2, -1])
        old_error = np.linalg.norm(kf.x[:2] - [13, 20])
        kf.update_pos(np.array([13., 20.]), [.1, .1])
        self.assertLess(np.linalg.norm(kf.x[:2] - [13, 20]), old_error)
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-12)
        self.assertTrue((np.linalg.eigvalsh(kf.P) >= 0).all())

    def test_optical_flow_known_translation_one_point(self):
        image = np.zeros((96, 96), dtype=np.uint8)
        image[35:50, 35:50] = 255
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), (96, 96))
        start = np.array([[35., 35.]])
        points, status = calc_optical_flow(image, shifted, start, start, 21, 2)
        self.assertEqual(points.shape, (1, 2))
        self.assertEqual(status.shape, (1,))
        self.assertTrue(status[0])
        np.testing.assert_allclose(points-start, [[3, 2]], atol=.15)

    def test_no_valid_flow_is_finite_zero(self):
        np.testing.assert_equal(calc_average_flow([[1, 2]], [[3, 4]], [False]), [0, 0])
        image = np.zeros((10, 10), dtype=np.uint8)
        points, status = calc_optical_flow(image, image, [], [], 5, 0)
        self.assertEqual(points.shape, (0, 2))
        self.assertEqual(status.shape, (0,))


class CounterTests(unittest.TestCase):
    def test_sequence_counts_two_with_shuffled_detections_and_noise(self):
        counter = DemoCounter()
        for detections in sequence():
            counter.step(detections)
        self.assertEqual(counter.finish(), 2)
        self.assertEqual(counter.finish(), 2)

    def test_active_track_counted_at_finish_only(self):
        counter = DemoCounter()
        for x in range(4):
            self.assertEqual(counter.step([[10+x, 10, 20, 20]]), 0)
        self.assertEqual(counter.finish(), 1)
        with self.assertRaises(RuntimeError):
            counter.step([])

    def test_one_frame_noise_discarded(self):
        counter = DemoCounter()
        counter.step([[10, 10, 20, 20]])
        counter.step([])
        self.assertEqual(counter.finish(), 0)

    def test_invalid_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            DemoCounter().step([[10, 10, -1, 20]])


if __name__ == "__main__":
    unittest.main()
