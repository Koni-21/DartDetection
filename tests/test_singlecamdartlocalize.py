import logging
import unittest

import numpy as np
from matplotlib import pyplot as plt

from dartdetect.singlecamdartlocalize import (
    filter_noise,
    compare_imgs,
    get_roi_coords,
    find_clusters,
    try_get_clusters_in_out,
    check_nr_of_clusters,
    lin_regression_on_cluster,
    dart_fully_arrived,
    dart_moved,
    overlap,
    differentiate_overlap,
    filter_cluster_by_usable_rows,
    calculate_position_from_cluster_and_image,
    occlusion_kind,
    check_overlap,
    calculate_position_from_occluded_dart,
    single_dart_removed,
    SingleCamLocalize,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


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
        pos, w0, w1, x, y = lin_regression_on_cluster(img, cluster)
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
        pos, w0, w1, x, y = lin_regression_on_cluster(img, cluster)
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
        pos, w0, w1, x, y = lin_regression_on_cluster(img, cluster)
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
        pos, w0, w1, x, y = lin_regression_on_cluster(img, cluster)
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
        pos, w0, w1, x, y = lin_regression_on_cluster(img, cluster)
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


class TestOverlap(unittest.TestCase):
    def test_overlap_no_overlap(self):
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        saved_dart_cluster = np.array([[3, 3], [4, 4], [5, 5]])
        expected_output = np.array([])
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

    def test_overlap_partial_overlap(self):
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        saved_dart_cluster = np.array([[1, 1], [3, 3], [4, 4]])
        expected_output = np.array([[1, 1]])
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

    def test_overlap_full_overlap(self):
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        saved_dart_cluster = np.array([[0, 0], [1, 1], [2, 2]])
        expected_output = np.array([[0, 0], [1, 1], [2, 2]])
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

    def test_overlap_empty_clusters(self):
        cluster = np.array([])
        saved_dart_cluster = np.array([])
        expected_output = np.array([])
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

    def test_overlap_one_empty_cluster(self):
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        saved_dart_cluster = np.array([])
        expected_output = np.array([])
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

        cluster = np.array([])
        saved_dart_cluster = np.array([[0, 0], [1, 1], [2, 2]])
        expected_output = np.array([])
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )


class TestDifferentiateOverlap(unittest.TestCase):
    def test_differentiate_overlap_partial_overlap(self):
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        overlap_points = np.array([[1, 1]])
        expected_output = {
            "fully_usable_rows": [0, 2],
            "middle_occluded_rows": [],
            "left_side_overlap": 1,
            "right_side_overlap": 0,
        }
        self.assertDictEqual(
            differentiate_overlap(cluster, overlap_points), expected_output
        )

    def test_differentiate_overlap_left_and_right_overlap(self):
        cluster = np.array([[0, 0], [1, 1], [2, 2], [1, 0], [2, 1]])
        overlap_points = np.array([[0, 0], [1, 1], [2, 2]])
        expected_output = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap": 1,
            "right_side_overlap": 2,
        }
        self.assertDictEqual(
            differentiate_overlap(cluster, overlap_points), expected_output
        )

    def test_differentiate_overlap_middle_overlap(self):
        cluster = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 23]])
        overlap_points = np.array([[0, 2], [1, 2]])
        expected_output = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [0, 1],
            "left_side_overlap": 0,
            "right_side_overlap": 0,
        }
        self.assertDictEqual(
            differentiate_overlap(cluster, overlap_points), expected_output
        )


class TestFilterClusterByUsableRows(unittest.TestCase):

    def test_filter_cluster_by_usable_rows_all_usable(self):
        cluster = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
        usable_rows = [0, 1]
        expected_output = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
        np.testing.assert_array_equal(
            filter_cluster_by_usable_rows(usable_rows, cluster), expected_output
        )

    def test_filter_cluster_by_usable_rows_some_usable(self):
        cluster = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]]
        )
        usable_rows = [0, 1]
        expected_output = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
        np.testing.assert_array_equal(
            filter_cluster_by_usable_rows(usable_rows, cluster), expected_output
        )


class TestCheckNrOfClusters(unittest.TestCase):
    def test_check_nr_of_clusters_no_clusters(self):
        clusters_in = []
        clusters_out = []
        expected_output = ([], [])
        self.assertEqual(
            check_nr_of_clusters(clusters_in, clusters_out), expected_output
        )

    def test_check_nr_of_clusters_single_incoming_cluster(self):
        clusters_in = [np.array([[0, 0], [1, 1]])]
        clusters_out = []
        expected_output = (clusters_in[0], [])
        self.assertEqual(
            check_nr_of_clusters(clusters_in, clusters_out), expected_output
        )

    def test_check_nr_of_clusters_single_outgoing_cluster(self):
        clusters_in = []
        clusters_out = [np.array([[0, 0], [1, 1]])]
        expected_output = ([], clusters_out[0])
        self.assertEqual(
            check_nr_of_clusters(clusters_in, clusters_out), expected_output
        )

    def test_check_nr_of_clusters_multiple_incoming_clusters(self):
        clusters_in = [np.array([[0, 0], [1, 1]]), np.array([[2, 2], [3, 3]])]
        clusters_out = []
        expected_output = (clusters_in[0], [])
        with self.assertLogs(LOGGER, level="WARNING") as log:
            self.assertEqual(
                check_nr_of_clusters(clusters_in, clusters_out), expected_output
            )
            self.assertIn("More than one new 'incoming' cluster found", log.output[0])

    def test_check_nr_of_clusters_multiple_outgoing_clusters(self):
        clusters_in = []
        clusters_out = [np.array([[0, 0], [1, 1]]), np.array([[2, 2], [3, 3]])]
        expected_output = ([], clusters_out[0])
        with self.assertLogs(LOGGER, level="WARNING") as log:
            self.assertEqual(
                check_nr_of_clusters(clusters_in, clusters_out), expected_output
            )
            self.assertIn("More than one new 'leaving' cluster found", log.output[0])

    def test_check_nr_of_clusters_multiple_incoming_and_outgoing_clusters(self):
        clusters_in = [np.array([[0, 0], [1, 1]]), np.array([[2, 2], [3, 3]])]
        clusters_out = [np.array([[4, 4], [5, 5]]), np.array([[6, 6], [7, 7]])]
        expected_output = (clusters_in[0], clusters_out[0])
        with self.assertLogs(LOGGER, level="WARNING") as log:
            self.assertEqual(
                check_nr_of_clusters(clusters_in, clusters_out), expected_output
            )
            self.assertIn("More than one new 'incoming' cluster found", log.output[0])
            self.assertIn("More than one new 'leaving' cluster found", log.output[1])


class TestCalculatePositionFromClusterAndImage(unittest.TestCase):

    def test_calculate_position_from_cluster_and_image_simple(self):
        img = np.array(
            [
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
        pos, angle_pred, support, r = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 1
        expected_angle_pred = 0
        expected_support = 4
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)

    def test_calculate_position_from_cluster_and_image_diagonal(self):
        img = np.array(
            [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
            ]
        )
        cluster = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        pos, angle_pred, support, r = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 3.0
        expected_angle_pred = -45.0
        expected_support = 4
        expected_r = 1.0
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(r, expected_r, places=2)

    def test_calculate_position_from_cluster_and_image_subpixel(self):
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
        pos, angle_pred, support, r = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 2.0
        expected_angle_pred = 0.0
        expected_support = 12

        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)

    def test_calculate_position_from_cluster_and_image_subpixel_2(self):
        img = np.array(
            [
                [1, 0.4, 0.4, 1, 1],
                [1, 0.5, 0.5, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2]])
        pos, angle_pred, support, r = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 1.5
        expected_angle_pred = 0.0
        expected_support = 4

        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)

    def test_calculate_position_from_cluster_and_image_subpixel_3(self):
        img = np.array(
            [
                [1, 0.2, 0.4, 1, 1],
                [1, 0.1, 0.5, 0.2, 1],
            ]
        )
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2], [1, 3]])
        pos, angle_pred, support, r = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 1.9545454545454548
        expected_angle_pred = -27.743204472006298
        expected_support = 5
        expected_r = 1.0
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(r, expected_r, places=2)


class TestOcclusionKind(unittest.TestCase):

    def test_occlusion_kind_fully_usable(self):
        occluded_rows = {
            "fully_usable_rows": [0, 1, 2],
            "middle_occluded_rows": [],
            "left_side_overlap": 0,
            "right_side_overlap": 0,
        }
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2), "fully_useable"
        )

    def test_occlusion_kind_middle_occluded(self):
        occluded_rows = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [0, 1, 2],
            "left_side_overlap": 0,
            "right_side_overlap": 0,
        }
        with self.assertRaises(NotImplementedError):
            occlusion_kind(occluded_rows, thresh_needed_rows=2)

    def test_occlusion_kind_one_side_fully_occluded(self):
        occluded_rows = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap": 3,
            "right_side_overlap": 0,
        }
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2),
            "one_side_fully_occluded",
        )

    def test_occlusion_kind_one_side_fully_occluded_right(self):
        occluded_rows = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap": 0,
            "right_side_overlap": 3,
        }
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2),
            "one_side_fully_occluded",
        )

    def test_occlusion_kind_not_enough_usable_rows(self):
        occluded_rows = {
            "fully_usable_rows": [0],
            "middle_occluded_rows": [],
            "left_side_overlap": 0,
            "right_side_overlap": 0,
        }
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2),
            "one_side_fully_occluded",
        )


class TestCheckOverlap(unittest.TestCase):

    def test_check_overlap_no_saved_darts(self):
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {}
        expected_output = {}
        result = check_overlap(cluster_in, saved_darts)
        self.assertDictEqual(result, expected_output)

    def test_check_overlap_no_overlap(self):
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[3, 3], [4, 4], [5, 5]])}}
        expected_output = {}
        self.assertDictEqual(check_overlap(cluster_in, saved_darts), expected_output)

    def test_check_overlap_partial_overlap(self):
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[1, 1], [3, 3], [4, 4]])}}
        expected_output = {
            "overlapping_darts": [1],
            "occlusion_kind": "one_side_fully_occluded",
            "overlap_points": np.array([[1, 1]]),
            "fully_usable_rows": [0, 2],
            "middle_occluded_rows": [],
            "left_side_overlap": 1,
            "right_side_overlap": 0,
        }
        result = check_overlap(cluster_in, saved_darts)
        for key in expected_output:
            if isinstance(expected_output[key], np.ndarray):
                np.testing.assert_array_equal(result[key], expected_output[key])
            else:
                self.assertEqual(result[key], expected_output[key])

    def test_check_overlap_full_overlap(self):
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[0, 0], [1, 1], [2, 2]])}}
        expected_output = {
            "overlapping_darts": [1],
            "occlusion_kind": "one_side_fully_occluded",
            "overlap_points": np.array([[0, 0], [1, 1], [2, 2]]),
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap": 3,
            "right_side_overlap": 0,
        }
        result = check_overlap(cluster_in, saved_darts)
        for key in expected_output:
            if isinstance(expected_output[key], np.ndarray):
                np.testing.assert_array_equal(result[key], expected_output[key])
            else:
                self.assertEqual(result[key], expected_output[key])

    def test_check_overlap_multiple_overlaps(self):
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {
            "d1": {"cluster": np.array([[0, 0], [1, 1]])},
            "d2": {"cluster": np.array([[2, 2], [3, 3]])},
        }
        expected_output = {
            "overlapping_darts": [1, 2],
            "occlusion_kind": "one_side_fully_occluded",
            "overlap_points": np.array([[1, 1], [0, 0], [2, 2]]),
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap": 3,
            "right_side_overlap": 0,
        }
        result = check_overlap(cluster_in, saved_darts)
        for key in expected_output:
            if isinstance(expected_output[key], np.ndarray):
                np.testing.assert_array_equal(result[key], expected_output[key])
            else:
                self.assertEqual(result[key], expected_output[key])

    # def test_check_overlap_middle_occlusion(self):
    #     cluster_in = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
    #     saved_darts = {"d1": {"cluster": np.array([[0, 2], [1, 2]])}}
    #     with self.assertRaises(NotImplementedError):
    #         check_overlap(cluster_in, saved_darts)


class TestCalculatePositionFromOccludedDart(unittest.TestCase):

    def test_calculate_position_from_occluded_dart_fully_usable(self):
        occlusion_dict = {
            "occlusion_kind": "fully_useable",
            "fully_usable_rows": [0, 1, 2],
        }
        cluster_in = np.array([[0, 1], [1, 1], [2, 1]])
        diff_img = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        current_img = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        saved_darts = {}

        pos, angle, support, r, error = calculate_position_from_occluded_dart(
            occlusion_dict, cluster_in, diff_img, current_img, saved_darts
        )

        expected_pos = 1
        expected_angle = 0
        expected_support = 3
        expected_error = 0

        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle, expected_angle, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(error, expected_error, places=2)

    def test_calculate_position_from_occluded_dart_one_side_fully_occluded(self):
        occlusion_dict = {
            "occlusion_kind": "one_side_fully_occluded",
            "overlapping_darts": [1],
        }
        cluster_in = np.array([[0, 1], [1, 1], [2, 1]])
        diff_img = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        current_img = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        saved_darts = {
            "d1": {
                "cluster": np.array([[0, 0], [1, 0], [2, 0]]),
                "img_pre": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            }
        }

        pos, angle, support, r, error = calculate_position_from_occluded_dart(
            occlusion_dict, cluster_in, diff_img, current_img, saved_darts
        )

        expected_pos = 0.75
        expected_angle = 0
        expected_support = 3
        expected_error = 0.125

        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle, expected_angle, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(error, expected_error, places=2)


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


class TestSingleCamLocalize(unittest.TestCase):
    def setUp(self):
        self.Loc = SingleCamLocalize()

    def test_new_image_single_image(self):
        img = np.ones((5, 5))
        result = self.Loc.new_image(img)
        self.assertIsNone(result)
        self.assertEqual(self.Loc.image_count, 1)
        self.assertEqual(len(self.Loc.imgs), 1)
        np.testing.assert_array_equal(self.Loc.current_img, img)

    def test_new_image_multiple_images(self):
        img1 = np.ones((5, 5))
        img2 = np.ones((5, 5)) * 0.5
        self.Loc.new_image(img1)
        result = self.Loc.new_image(img2)
        self.assertIsNotNone(result)
        self.assertEqual(self.Loc.image_count, 2)
        self.assertEqual(len(self.Loc.imgs), 2)
        np.testing.assert_array_equal(self.Loc.current_img, img2)

    def test_analyse_imgs_no_change(self):
        img1 = np.ones((5, 5))
        img2 = np.ones((5, 5))
        self.Loc.new_image(img1)
        result = self.Loc.new_image(img2)
        self.assertIsNone(result)

    def test_incoming_cluster_detected(self):
        img1 = np.ones((3, 5))
        img2 = np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
            ]
        )
        self.Loc.new_image(img1)
        result = self.Loc.new_image(img2)
        self.assertIsNotNone(result)
        self.assertIn("pos", result)
        self.assertEqual(2, result["pos"])
        self.assertIn("angle", result)
        self.assertIn("r", result)
        self.assertIn("support", result)
        self.assertIn("error", result)

    def test_leaving_cluster_not_arrived_detected(self):
        img1 = np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        img2 = np.ones((5, 5))
        self.Loc.new_image(img1)
        result = self.Loc.new_image(img2)
        self.assertIsNone(result)
        self.assertEqual(len(self.Loc.saved_darts), 0)

    def test_incoming_and_leaving_cluster_detected(self):
        img1 = np.ones((3, 5))
        img2 = np.array(
            [
                [1, 1, 0.4, 0, 1],
                [1, 1, 0.5, 0.1, 1],
                [1, 1, 0.2, 0.4, 1],
            ]
        )
        img3 = np.ones((3, 5))
        self.Loc.new_image(img1)
        result_incoming = self.Loc.new_image(img2)

        self.assertIsNotNone(result_incoming)
        self.assertEqual(len(self.Loc.saved_darts), 1)

        result_leaving = self.Loc.new_image(img3)

        self.assertIsNone(result_leaving)
        self.assertEqual(len(self.Loc.saved_darts), 0)

    def test_visualize_stream(self):
        img1 = np.ones((3, 5))
        img2 = np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
            ]
        )
        self.Loc.new_image(img1)
        self.Loc.new_image(img2)
        fig, ax = plt.subplots()
        self.Loc.visualize_stream(ax)
        self.assertEqual(len(ax.lines), 1)


if __name__ == "__main__":
    unittest.main()
