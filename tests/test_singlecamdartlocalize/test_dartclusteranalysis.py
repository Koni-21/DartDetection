import unittest
import pytest

import numpy as np

from dartdetect.singlecamdartlocalize.dartclusteranalysis import (
    filter_noise,
    compare_imgs,
    get_roi_coords,
    find_clusters,
    try_get_clusters_in_out,
    lin_regression_on_cluster,
    calculate_position_from_cluster_and_image,
)


class TestFilterNoise(unittest.TestCase):

    def test_filter_noise(self):
        img = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        thresh = 0.1
        expected_output = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 1.0]])
        np.testing.assert_array_equal(filter_noise(img, thresh), expected_output)


class TestCompareImgs(unittest.TestCase):

    def test_compare_imgs_emtpy(self):
        previous_img = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        current_img = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

        expected_output = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(
            compare_imgs(previous_img, current_img), expected_output
        )

    def test_compare_imgs_new_dart(self):
        previous_img = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        current_img = np.array([[1, 1, 0.4, 0, 1], [1, 1, 0.5, 0.1, 1]])

        expected_output = np.array([[1, 1, 0.4, 0, 1], [1, 1, 0.5, 0.1, 1]])
        np.testing.assert_array_almost_equal(
            compare_imgs(previous_img, current_img), expected_output
        )

    def test_compare_imgs_moved_dart(self):
        # is the 1.6 as difference the desired solution for a leaving dart?
        previous_img = np.array([[1, 1, 0.4, 0, 1], [1, 1, 0.5, 0.1, 1]])
        current_img = np.array([[1, 1, 1.0, 0.5, 0.4], [1, 1, 1, 0.6, 0.3]])

        expected_output = np.array([[1, 1, 1.6, 1.5, 0.4], [1, 1, 1.5, 1.5, 0.3]])
        np.testing.assert_array_almost_equal(
            compare_imgs(previous_img, current_img), expected_output
        )


class TestGetRoiCoords(unittest.TestCase):

    def test_get_roi_coords_no_change(self):
        diff_img = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        expected_output = 0
        np.testing.assert_array_equal(len(get_roi_coords(diff_img)), expected_output)

    def test_get_roi_coords_incoming_dart(self):
        diff_img = np.array([[1, 1, 0.4, 0, 1], [1, 1, 0.5, 0.1, 1]])
        expected_output = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
        np.testing.assert_array_equal(get_roi_coords(diff_img), expected_output)

    def test_get_roi_coords_leaving_dart(self):
        diff_img = np.array([[1, 1, 1.6, 1.5, 1], [1, 1, 1.5, 1.5, 1]])
        expected_output = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
        np.testing.assert_array_equal(
            get_roi_coords(diff_img, incoming=False), expected_output
        )

    def test_get_roi_coords_mixed(self):
        diff_img = np.array([[1, 0.8, 1.6, 0.5, 1], [1, 1, 0.5, 1.5, 0.3]])
        expected_output_incoming = np.array([[0, 1], [0, 3], [1, 2], [1, 4]])
        expected_output_leaving = np.array([[0, 2], [1, 3]])
        np.testing.assert_array_equal(
            get_roi_coords(diff_img), expected_output_incoming
        )
        np.testing.assert_array_equal(
            get_roi_coords(diff_img, incoming=False), expected_output_leaving
        )


class TestFindClusters(unittest.TestCase):

    def test_find_clusters_single_cluster(self):
        coordinates = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
        thresh_n_pixels_dart = 3
        expected_output = [np.array([[0, 0], [0, 1], [1, 0], [1, 1]])]
        result = find_clusters(coordinates, thresh_n_pixels_dart)
        self.assertEqual(len(result), len(expected_output))
        for res, exp in zip(result, expected_output):
            np.testing.assert_array_equal(res, exp)

    def test_find_clusters_multiple_clusters(self):
        coordinates = np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11]]
        )
        thresh_n_pixels_dart = 3
        expected_output = [
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            np.array([[10, 10], [10, 11], [11, 10], [11, 11]]),
        ]
        result = find_clusters(coordinates, thresh_n_pixels_dart)
        self.assertEqual(len(result), len(expected_output))
        for res, exp in zip(result, expected_output):
            np.testing.assert_array_equal(res, exp)

    def test_find_clusters_below_threshold(self):
        coordinates = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
        thresh_n_pixels_dart = 5
        expected_output = []
        result = find_clusters(coordinates, thresh_n_pixels_dart)
        self.assertEqual(result, expected_output)


class TestTryGetClustersInOut(unittest.TestCase):
    def test_try_get_clusters_in_out_no_change(self):
        diff_img = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        expected_clusters_in = []
        expected_clusters_out = []
        clusters_in, clusters_out = try_get_clusters_in_out(diff_img)
        self.assertEqual(clusters_in, expected_clusters_in)
        self.assertEqual(clusters_out, expected_clusters_out)

    def test_try_get_clusters_in_out_incoming_dart(self):
        diff_img = np.array([[1, 1, 0.4, 0, 1], [1, 1, 0.5, 0.1, 1]])
        expected_clusters_in = [np.array([[0, 2], [0, 3], [1, 2], [1, 3]])]
        expected_clusters_out = []
        clusters_in, clusters_out = try_get_clusters_in_out(diff_img)
        self.assertEqual(len(clusters_in), len(expected_clusters_in))
        for res, exp in zip(clusters_in, expected_clusters_in):
            np.testing.assert_array_equal(res, exp)
        self.assertEqual(clusters_out, expected_clusters_out)

    def test_try_get_clusters_in_out_leaving_dart(self):
        diff_img = np.array([[1, 1, 1.6, 1.5, 1], [1, 1, 1.5, 1.5, 1]])
        expected_clusters_in = []
        expected_clusters_out = [np.array([[0, 2], [0, 3], [1, 2], [1, 3]])]
        clusters_in, clusters_out = try_get_clusters_in_out(diff_img)
        self.assertEqual(clusters_in, expected_clusters_in)
        self.assertEqual(len(clusters_out), len(expected_clusters_out))
        for res, exp in zip(clusters_out, expected_clusters_out):
            np.testing.assert_array_equal(res, exp)

    def test_try_get_clusters_in_out_below_threshold(self):
        diff_img = np.array([[1, 1, 0.9, 0.8, 1], [1, 1, 0.9, 0.8, 1]])
        expected_clusters_in = []
        expected_clusters_out = []
        clusters_in, clusters_out = try_get_clusters_in_out(
            diff_img, thresh_n_pixels_dart=5
        )
        self.assertEqual(clusters_in, expected_clusters_in)
        self.assertEqual(clusters_out, expected_clusters_out)


class TestLinRegressionOnCluster(unittest.TestCase):

    def test_lin_regression_on_cluster_simple(self):
        img = np.array(
            [
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
        pos, w0, w1, x, y, error = lin_regression_on_cluster(img, cluster)
        expected_pos = 1
        expected_w0 = 1
        expected_w1 = 0
        expected_x = [0, 1, 2, 3]
        expected_y = [1, 1, 1, 1]
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(w0, expected_w0, places=2)
        self.assertAlmostEqual(w1, expected_w1, places=2)
        self.assertListEqual(x, expected_x)
        self.assertListEqual(y, expected_y)

    def test_lin_regression_on_cluster_diagonal(self):
        img = np.array(
            [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
            ]
        )

        cluster = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        pos, w0, w1, x, y, error = lin_regression_on_cluster(img, cluster)
        expected_pos = 3.0
        expected_w0 = 0
        expected_w1 = 1.0
        expected_x = [0, 1, 2, 3]
        expected_y = [0, 1, 2, 3]
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(w0, expected_w0, places=2)
        self.assertAlmostEqual(w1, expected_w1, places=2)
        self.assertListEqual(x, expected_x)
        self.assertListEqual(y, expected_y)

    def test_lin_regression_on_cluster_subpixel(self):
        img = np.array(
            [
                [1, 0.3, 0, 0.3, 1],
                [1, 0.4, 0, 0.4, 1],
                [1, 0.3, 0, 0.3, 1],
                [1, 0.3, 0, 0.3, 1],
            ]
        )
        cluster = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, 1],
                [2, 2],
                [2, 3],
                [3, 1],
                [3, 2],
                [3, 3],
            ]
        )
        pos, w0, w1, x, y, error = lin_regression_on_cluster(img, cluster)
        expected_pos = 2.0
        expected_w0 = 2.0
        expected_w1 = 0.0
        expected_x = [0, 1, 2, 3]
        expected_y = [2, 2, 2, 2]
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(w0, expected_w0, places=2)
        self.assertAlmostEqual(w1, expected_w1, places=2)
        self.assertListEqual(x, expected_x)
        self.assertListEqual(y, expected_y)

    def test_lin_regression_on_cluster_subpixel_2(self):
        img = np.array(
            [
                [1, 0.4, 0.4, 1, 1],
                [1, 0.5, 0.5, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2]])
        pos, w0, w1, x, y, error = lin_regression_on_cluster(img, cluster)
        expected_pos = 1.5
        expected_w0 = 1.5
        expected_w1 = 0.0
        expected_x = [0, 1]
        expected_y = [1.5, 1.5]
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(w0, expected_w0, places=2)
        self.assertAlmostEqual(w1, expected_w1, places=2)
        self.assertListEqual(x, expected_x)
        self.assertListEqual(y, expected_y)

    def test_lin_regression_on_cluster_subpixel_3(self):
        img = np.array(
            [
                [1, 0.2, 0.4, 1, 1],
                [1, 0.1, 0.5, 0.2, 1],
            ]
        )
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2], [1, 3]])
        pos, w0, w1, x, y, error = lin_regression_on_cluster(img, cluster)
        expected_pos = 1.9545454545454548
        expected_w0 = 1.4285714285714286
        expected_w1 = 0.5259740259740262
        expected_x = [0, 1]
        expected_y = [1.4285714285714286, 1.9545454545454548]
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(w0, expected_w0, places=2)
        self.assertAlmostEqual(w1, expected_w1, places=2)
        self.assertListEqual(x, expected_x)
        self.assertListEqual(y, expected_y)

    def test_lin_regression_on_cluster_simple_not_weighted(self):
        img = np.array(
            [
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
        pos, w0, w1, x, y, error = lin_regression_on_cluster(
            img, cluster, weighted=False
        )
        expected_pos = 1
        expected_w0 = 1
        expected_w1 = 0
        expected_x = [0, 1, 2, 3]
        expected_y = [1, 1, 1, 1]
        expected_error = 0.5
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(w0, expected_w0, places=2)
        self.assertAlmostEqual(w1, expected_w1, places=2)
        self.assertListEqual(x, expected_x)
        self.assertListEqual(y, expected_y)
        self.assertAlmostEqual(error, expected_error)

    def test_lin_regression_on_cluster_subpixel_3_not_weighted(self):
        img = np.array(
            [
                [1, 0.2, 0.4, 1, 1],
                [1, 0.1, 0.5, 0.2, 1],
            ]
        )
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2], [1, 3]])
        pos, w0, w1, x, y, error = lin_regression_on_cluster(
            img, cluster, weighted=False
        )
        expected_pos = 2
        expected_w0 = 1.5
        expected_w1 = 0.5
        expected_x = [0, 1]
        expected_y = [1.5, 2]
        expected_error = 0.5
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(w0, expected_w0, places=2)
        self.assertAlmostEqual(w1, expected_w1, places=2)
        self.assertListEqual(x, expected_x)
        self.assertListEqual(y, expected_y)
        self.assertAlmostEqual(error, expected_error)


class TestCalculatePositionFromClusterAndImage(unittest.TestCase):

    def test_calculate_position_simple(self):
        img = np.array(
            [
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 1.0
        expected_angle_pred = 0.0
        expected_support = 4
        expected_r = 1.0
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(r, expected_r, places=2)

    def test_calculate_position_diagonal(self):
        img = np.array(
            [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
            ]
        )
        cluster = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 3.0
        expected_angle_pred = -45.0
        expected_support = 4
        expected_r = 1
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(r, expected_r, places=2)

    def test_calculate_position_not_weighted(self):
        img = np.array(
            [
                [1, 0.2, 0.4, 1, 1],
                [1, 0.1, 0.5, 0.2, 1],
            ]
        )
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2], [1, 3]])
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster, weighted=False
        )
        expected_pos = 2.0
        expected_angle_pred = -26.57
        expected_support = 2
        expected_r = 1.0
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(r, expected_r, places=2)
