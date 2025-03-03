import unittest
import pytest

import numpy as np

from dartdetect.singlecamdartlocalize.dartmovementutils import (
    dart_fully_arrived,
    dart_moved,
    single_dart_removed,
)


class TestDartFullyArrived(unittest.TestCase):
    def test_dart_fully_arrived_true(self):
        img_height = 10
        cluster = np.array([[8, 1], [9, 2]])
        self.assertTrue(dart_fully_arrived(img_height, cluster))

    def test_dart_fully_arrived_false(self):
        img_height = 10
        cluster = np.array([[7, 1], [8, 2]])
        self.assertFalse(dart_fully_arrived(img_height, cluster))

    def test_dart_fully_arrived_with_distance_to_bottom(self):
        img_height = 10
        cluster = np.array([[7, 1], [8, 2]])
        distance_to_bottom = 2
        self.assertTrue(dart_fully_arrived(img_height, cluster, distance_to_bottom))


class TestDartMoved(unittest.TestCase):
    def test_dart_moved_true(self):
        diff_img = np.array([[1, 1.6, 0, 0.6, 1], [1, 1.4, 0, 0.4, 1]])
        cluster_in = np.array([[0, 3], [1, 3]])
        cluster_out = np.array([[0, 1], [1, 1]])
        self.assertTrue(
            dart_moved(diff_img, cluster_in, cluster_out, difference_thresh=0)
        )

    def test_dart_moved_no_outgoing_cluster(self):
        diff_img = np.array([[1, 1, 0, 0.5, 1], [1, 0, 0.5, 1, 1]])
        cluster_in = np.array([[0, 4], [1, 4]])
        cluster_out = None
        self.assertFalse(dart_moved(diff_img, cluster_in, cluster_out))

    def test_dart_moved_dart_removed(self):
        diff_img = np.array([[0.7, 1, 2.0, 2.0, 2.0], [1, 1, 2.0, 2.0, 2.0]])
        cluster_in = np.array([[0, 0]])
        cluster_out = np.array([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]])
        self.assertFalse(dart_moved(diff_img, cluster_in, cluster_out))


class TestSingleDartRemoved(unittest.TestCase):
    def test_single_dart_removed_no_dart(self):
        diff_img = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        cluster_out = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {}
        self.assertIsNone(single_dart_removed(diff_img, cluster_out, saved_darts))

    def test_single_dart_removed_dart_removed(self):
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = np.array([[0, 2], [1, 2], [2, 2]])
        saved_darts = {"d1": {"pos": 2}}
        self.assertEqual(single_dart_removed(diff_img, cluster_out, saved_darts), "d1")

    def test_single_dart_removed_not_yet_detected_dart_removed(self):
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = np.array([[0, 2], [1, 2], [2, 2]])
        saved_darts = {"d1": {"pos": 0}}
        self.assertIsNone(single_dart_removed(diff_img, cluster_out, saved_darts))

    def test_single_dart_removed_with_tolerance(self):
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = np.array([[0, 2], [1, 2], [2, 2]])
        saved_darts = {"d1": {"pos": 1.5}}
        self.assertEqual(
            single_dart_removed(diff_img, cluster_out, saved_darts, tolerance_px=1),
            "d1",
        )

    def test_single_dart_removed_multiple_darts(self):
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = np.array([[0, 2], [1, 2], [2, 2]])
        saved_darts = {"d1": {"pos": 1}, "d2": {"pos": 2}}
        self.assertEqual(single_dart_removed(diff_img, cluster_out, saved_darts), "d2")
