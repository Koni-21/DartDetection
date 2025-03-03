import unittest
import unittest.mock
import pytest

import numpy as np
import io
import sys

from dartdetect.singlecamdartlocalize.moveddartocclusion import (
    _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row,
    calculate_angle_of_different_clusters,
    combine_clusters_based_on_the_angle,
)


class TestReduceCluster(unittest.TestCase):
    def test_reduce_cluster_left_true(self):
        # Test keeping rightmost pixel when left=True
        cluster = np.array([[1, 5], [1, 8], [2, 3], [2, 6], [3, 4]])
        expected = np.array([[1, 8], [2, 6], [3, 4]])
        result = _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row(
            cluster, left=True
        )
        np.testing.assert_array_equal(result, expected)

    def test_reduce_cluster_left_false(self):
        # Test keeping leftmost pixel when left=False
        cluster = np.array([[1, 5], [1, 8], [2, 3], [2, 6], [3, 4]])
        expected = np.array([[1, 5], [2, 3], [3, 4]])
        result = _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row(
            cluster, left=False
        )
        np.testing.assert_array_equal(result, expected)

    def test_reduce_cluster_single_point(self):
        # Test with a single point
        cluster = np.array([[5, 10]])
        expected = np.array([[5, 10]])
        result = _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row(cluster)
        np.testing.assert_array_equal(result, expected)

    def test_reduce_cluster_same_row_different_columns(self):
        # Test with multiple points all on same row
        cluster = np.array([[7, 1], [7, 3], [7, 5], [7, 2]])
        # Should keep only maximum column for left=True
        expected_left = np.array([[7, 5]])
        result = _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row(
            cluster, left=True
        )
        np.testing.assert_array_equal(result, expected_left)

        # Should keep only minimum column for left=False
        expected_right = np.array([[7, 1]])
        result = _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row(
            cluster, left=False
        )
        np.testing.assert_array_equal(result, expected_right)


class TestCalculateAngleOfDifferentClusters(unittest.TestCase):
    @unittest.mock.patch(
        "dartdetect.singlecamdartlocalize.moveddartocclusion.calculate_position_from_cluster_and_image"
    )
    @unittest.mock.patch(
        "dartdetect.singlecamdartlocalize.moveddartocclusion.filter_cluster_by_usable_rows"
    )
    def test_calculate_angle_different_clusters(self, mock_filter, mock_calc_pos):
        # Setup mock returns
        mock_filter.return_value = np.array([[5, 1], [6, 2], [7, 3]])
        mock_calc_pos.side_effect = [
            (None, 30.0, None, None, None),  # First cluster: 30 degrees
            (None, 45.0, None, None, None),  # Second cluster: 45 degrees
            (None, 60.0, None, None, None),  # Third cluster: 60 degrees
        ]

        # Create test data
        diff_img = np.ones((10, 10))
        clusters = [
            np.array([[1, 5], [1, 8], [2, 6]]),  # right_side_fully_occluded
            np.array([[3, 2], [3, 7], [4, 5]]),  # left_side_fully_occluded
            np.array([[5, 1], [6, 2], [7, 3]]),  # fully_usable
        ]

        which_side_overlap = {
            0: "right_side_fully_occluded",
            1: "left_side_fully_occluded",
            2: "fully_usable",
        }

        occluded_rows_clusters = {0: [], 1: [], 2: [5, 6, 7]}

        # Call the function (suppress print output)
        captured_output = io.StringIO()
        sys.stdout = captured_output
        result = calculate_angle_of_different_clusters(
            diff_img, clusters, which_side_overlap, occluded_rows_clusters
        )
        sys.stdout = sys.__stdout__

        # Verify results
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 30.0)
        self.assertEqual(result[1], 45.0)
        self.assertEqual(result[2], 60.0)

        # Verify filter was called correctly for fully_usable cluster
        mock_filter.assert_called_once_with(occluded_rows_clusters[2], clusters[2])


class TestCombineClustersBasedOnTheAngle(unittest.TestCase):
    def test_combine_clusters_based_on_the_angle(self):
        # Create test clusters
        clusters = [
            np.array([[1, 1], [2, 2]]),
            np.array([[3, 3], [4, 4]]),
            np.array([[5, 5], [6, 6]]),
        ]

        # Angles that will cause clusters 0 and 2 to be combined (within 5 degrees)
        angle_of_clusters = {0: 30.0, 1: 45.0, 2: 32.0}  # Within 5 degrees of cluster 0

        # Call the function
        result = combine_clusters_based_on_the_angle(clusters, angle_of_clusters)

        # There should be 1 combined cluster (clusters 0 and 2)
        self.assertEqual(len(result), 1)

        # The combined cluster should contain all points from clusters 0 and 2
        expected_combined = np.vstack([clusters[0], clusters[2]])
        np.testing.assert_array_equal(result[0], expected_combined)

    def test_no_clusters_with_similar_angles(self):
        # Test when no clusters have similar angles
        clusters = [
            np.array([[1, 1], [2, 2]]),
            np.array([[3, 3], [4, 4]]),
            np.array([[5, 5], [6, 6]]),
        ]
        # All angles differ by more than 5 degrees
        angle_of_clusters = {0: 30.0, 1: 40.0, 2: 50.0}

        result = combine_clusters_based_on_the_angle(clusters, angle_of_clusters)

        # Should return empty list since no clusters have similar angles
        self.assertEqual(len(result), 0)
        self.assertEqual(result, [])

    def test_multiple_pairs_with_similar_angles(self):
        # Test when multiple pairs of clusters have similar angles
        clusters = [
            np.array([[1, 1], [2, 2]]),
            np.array([[3, 3], [4, 4]]),
            np.array([[5, 5], [6, 6]]),
            np.array([[7, 7], [8, 8]]),
        ]
        # Clusters 0 and 2 are similar, and clusters 1 and 3 are similar
        angle_of_clusters = {0: 30.0, 1: 60.0, 2: 32.0, 3: 62.0}

        result = combine_clusters_based_on_the_angle(clusters, angle_of_clusters)

        # Should return 2 combined clusters
        self.assertEqual(len(result), 2)

        # Check combinations are correct (order might vary)
        expected_combinations = [
            np.vstack([clusters[0], clusters[2]]),
            np.vstack([clusters[1], clusters[3]]),
        ]

        for combined in result:
            self.assertTrue(
                any(
                    np.array_equal(combined, expected)
                    for expected in expected_combinations
                )
            )

    def test_three_clusters_with_similar_angles(self):
        # Test when three clusters all have similar angles
        clusters = [
            np.array([[1, 1], [2, 2]]),
            np.array([[3, 3], [4, 4]]),
            np.array([[5, 5], [6, 6]]),
        ]
        # All clusters have angles within 5 degrees of each other
        angle_of_clusters = {0: 30.0, 1: 32.0, 2: 34.0}

        result = combine_clusters_based_on_the_angle(clusters, angle_of_clusters)

        # Should return 3 combined clusters (0+1, 1+2, 0+2)
        self.assertEqual(len(result), 3)

        expected_combinations = [
            np.vstack([clusters[0], clusters[1]]),
            np.vstack([clusters[1], clusters[2]]),
            np.vstack([clusters[0], clusters[2]]),
        ]

        for combined in result:
            self.assertTrue(
                any(
                    np.array_equal(combined, expected)
                    for expected in expected_combinations
                )
            )

    def test_empty_clusters_list(self):
        # Test with empty clusters list
        clusters = []
        angle_of_clusters = {}

        result = combine_clusters_based_on_the_angle(clusters, angle_of_clusters)

        # Should return empty list
        self.assertEqual(result, [])

    def test_single_cluster(self):
        # Test with just one cluster
        clusters = [np.array([[1, 1], [2, 2]])]
        angle_of_clusters = {0: 45.0}

        result = combine_clusters_based_on_the_angle(clusters, angle_of_clusters)

        # Should return empty list (can't combine a single cluster)
        self.assertEqual(result, [])
