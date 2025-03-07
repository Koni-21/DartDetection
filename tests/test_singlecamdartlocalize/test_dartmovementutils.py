import unittest
import pytest

import numpy as np
from typing import Dict, Any

from dartdetect.singlecamdartlocalize.dartmovementutils import (
    dart_fully_arrived,
    dart_moved,
    single_dart_removed,
    _find_matching_dart,
)


class TestDartFullyArrived(unittest.TestCase):
    """Tests for the dart_fully_arrived function."""

    def test_dart_fully_arrived_true(self):
        """Test when dart is at the bottom of the frame."""
        img_height = 10
        cluster = np.array([[8, 1], [9, 2]])
        self.assertTrue(dart_fully_arrived(img_height, cluster))

    def test_dart_fully_arrived_false(self):
        """Test when dart is not at the bottom of the frame."""
        img_height = 10
        cluster = np.array([[7, 1], [8, 2]])
        self.assertFalse(dart_fully_arrived(img_height, cluster))

    def test_dart_fully_arrived_with_distance_to_bottom(self):
        """Test with custom distance_to_bottom parameter."""
        img_height = 10
        cluster = np.array([[7, 1], [8, 2]])
        distance_to_bottom = 2
        self.assertTrue(dart_fully_arrived(img_height, cluster, distance_to_bottom))

    def test_dart_fully_arrived_edge_case(self):
        """Test edge case with dart exactly at threshold."""
        img_height = 10
        distance_to_bottom = 2
        # Maximum row is exactly at the threshold
        cluster = np.array([[6, 1], [8, 2]])  # max_row=8, threshold=8
        self.assertTrue(dart_fully_arrived(img_height, cluster, distance_to_bottom))


class TestDartMoved(unittest.TestCase):
    """Tests for the dart_moved function."""

    def test_dart_moved_true(self):
        """Test when dart has moved."""
        diff_img = np.array([[1, 1.6, 0, 0.6, 1], [1, 1.4, 0, 0.4, 1]])
        cluster_in = np.array([[0, 3], [1, 3]])
        cluster_out = np.array([[0, 1], [1, 1]])
        self.assertTrue(
            dart_moved(diff_img, cluster_in, cluster_out, difference_thresh=0)
        )

    def test_dart_moved_no_outgoing_cluster(self):
        """Test when there's no outgoing cluster."""
        diff_img = np.array([[1, 1, 0, 0.5, 1], [1, 0, 0.5, 1, 1]])
        cluster_in = np.array([[0, 4], [1, 4]])
        cluster_out = None
        self.assertFalse(dart_moved(diff_img, cluster_in, cluster_out))

    def test_dart_moved_dart_removed(self):
        """Test when dart is removed rather than moved."""
        diff_img = np.array([[0.7, 1, 2.0, 2.0, 2.0], [1, 1, 2.0, 2.0, 2.0]])
        cluster_in = np.array([[0, 0]])
        cluster_out = np.array([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]])
        self.assertFalse(dart_moved(diff_img, cluster_in, cluster_out))

    def test_dart_moved_with_custom_threshold(self):
        """Test with custom difference threshold."""
        diff_img = np.array([[1, 1.6, 0, 0.2, 1], [1, 1.4, 0, 0.1, 1]])
        cluster_in = np.array([[0, 3], [1, 3]])
        cluster_out = np.array([[0, 1], [1, 1]])
        # With threshold 5, the difference is within limits
        self.assertTrue(
            dart_moved(diff_img, cluster_in, cluster_out, difference_thresh=1)
        )
        # With threshold 0.1, the difference is outside limits
        self.assertFalse(
            dart_moved(diff_img, cluster_in, cluster_out, difference_thresh=0.1)
        )


class TestSingleDartRemoved(unittest.TestCase):
    """Tests for the single_dart_removed function."""

    def test_single_dart_removed_no_dart(self):
        """Test when no saved dart matches the removed position."""
        diff_img = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        cluster_out = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {}
        self.assertIsNone(single_dart_removed(diff_img, cluster_out, saved_darts))

    def test_single_dart_removed_dart_removed(self):
        """Test when a dart is successfully detected as removed."""
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = np.array([[0, 2], [1, 2], [2, 2]])
        saved_darts = {"d1": {"pos": 2}}
        self.assertEqual(single_dart_removed(diff_img, cluster_out, saved_darts), "d1")

    def test_single_dart_removed_not_yet_detected_dart_removed(self):
        """Test when the removed dart position doesn't match any saved dart."""
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = np.array([[0, 2], [1, 2], [2, 2]])
        saved_darts = {"d1": {"pos": 0}}
        self.assertIsNone(single_dart_removed(diff_img, cluster_out, saved_darts))

    def test_single_dart_removed_with_tolerance(self):
        """Test with custom tolerance for position matching."""
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = np.array([[0, 2], [1, 2], [2, 2]])
        saved_darts = {"d1": {"pos": 1.5}}
        self.assertEqual(
            single_dart_removed(diff_img, cluster_out, saved_darts, tolerance_px=1),
            "d1",
        )

    def test_single_dart_removed_multiple_darts(self):
        """Test when multiple saved darts exist but only one matches."""
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = np.array([[0, 2], [1, 2], [2, 2]])
        saved_darts = {"d1": {"pos": 1}, "d2": {"pos": 2}}
        self.assertEqual(
            single_dart_removed(diff_img, cluster_out, saved_darts, tolerance_px=0.5),
            "d2",
        )

    def test_single_dart_removed_empty_cluster(self):
        """Test when cluster_out is empty."""
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = np.array([])
        saved_darts = {"d1": {"pos": 2}}
        self.assertIsNone(single_dart_removed(diff_img, cluster_out, saved_darts))

    def test_single_dart_removed_none_cluster(self):
        """Test when cluster_out is None."""
        diff_img = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])
        cluster_out = None
        saved_darts = {"d1": {"pos": 2}}
        self.assertIsNone(single_dart_removed(diff_img, cluster_out, saved_darts))


class TestFindMatchingDart(unittest.TestCase):
    """Tests for the _find_matching_dart helper function."""

    def test_find_matching_dart_found(self):
        """Test when a matching dart is found."""
        saved_darts = {"d1": {"pos": 10}, "d2": {"pos": 20}}
        position = 10.5
        tolerance_px = 1
        self.assertEqual(_find_matching_dart(saved_darts, position, tolerance_px), "d1")

    def test_find_matching_dart_not_found(self):
        """Test when no matching dart is found."""
        saved_darts = {"d1": {"pos": 10}, "d2": {"pos": 20}}
        position = 15
        tolerance_px = 1
        self.assertIsNone(_find_matching_dart(saved_darts, position, tolerance_px))

    def test_find_matching_dart_exact_match(self):
        """Test when position exactly matches a dart."""
        saved_darts = {"d1": {"pos": 10}, "d2": {"pos": 20}}
        position = 10.0
        tolerance_px = 0.0
        self.assertEqual(_find_matching_dart(saved_darts, position, tolerance_px), "d1")

    def test_find_matching_dart_empty_dict(self):
        """Test with empty saved_darts dictionary."""
        saved_darts: Dict[str, Dict[str, float]] = {}
        position = 10
        self.assertIsNone(_find_matching_dart(saved_darts, position))
