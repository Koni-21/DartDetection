import unittest
import pytest

import numpy as np

from dartdetect.singlecamdartlocalize.dartocclusionhandler import (
    dilate_cluster,
    overlap,
    differentiate_overlap,
    filter_cluster_by_usable_rows,
    filter_middle_overlap_combined_cluster,
    calculate_position_from_cluster_and_image,
    check_overlap,
    occlusion_kind,
    check_occlusion_type_of_a_single_cluster,
    calculate_position_from_occluded_dart,
    check_which_sides_are_occluded_of_the_clusters,
)


class TestDilateCluster(unittest.TestCase):
    def test_dilate_cluster_no_dilation(self):
        cluster_mask = np.array([[0, 1], [1, 1], [2, 1]])
        img_width = 5
        dilate_cluster_by_n_px = 0
        expected_output = np.array([[0, 1], [1, 1], [2, 1]])
        result = dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px)
        np.testing.assert_array_equal(result, expected_output)

    def test_dilate_cluster_single_pixel_dilation(self):
        cluster_mask = np.array([[0, 1], [1, 1], [2, 1]])
        img_width = 5
        dilate_cluster_by_n_px = 1
        expected_output = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )
        result = dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px)
        np.testing.assert_array_equal(result, expected_output)

    def test_dilate_cluster_multiple_pixel_dilation(self):
        cluster_mask = np.array([[0, 1], [1, 1], [2, 1]])
        img_width = 5
        dilate_cluster_by_n_px = 2
        expected_output = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, 0],
                [2, 1],
                [2, 2],
                [2, 3],
            ]
        )
        result = dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px)
        np.testing.assert_array_equal(result, expected_output)

    def test_dilate_cluster_boundary_conditions(self):
        cluster_mask = np.array([[0, 0], [1, 0], [2, 0]])
        img_width = 3
        dilate_cluster_by_n_px = 1
        expected_output = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
        result = dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px)
        np.testing.assert_array_equal(result, expected_output)


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
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [1],
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
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [1, 2],
            "single_pixel_thick_overlap_rows": [0],
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
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
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


class TestFilterMiddleOverlapCombinedCluster(unittest.TestCase):

    def test_filter_middle_overlap_combined_cluster_standard(self):
        middle_occluded_rows = [0, 1]
        overlap_points = np.array([[0, 2], [1, 2]])
        combined_cluster = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
        expected_output = np.array([[0, 1], [0, 3], [1, 1], [1, 3]])
        result = filter_middle_overlap_combined_cluster(
            middle_occluded_rows, overlap_points, combined_cluster
        )
        np.testing.assert_array_equal(result, expected_output)

    def test_filter_middle_overlap_combined_cluster_cutoff_not_symetric_part(self):
        middle_occluded_rows = [0, 1]
        overlap_points = np.array([[0, 2], [1, 2]])
        combined_cluster = np.array(
            [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 1], [1, 2], [1, 3]]
        )
        expected_output = np.array([[0, 1], [0, 5], [1, 1], [1, 3]])
        result = filter_middle_overlap_combined_cluster(
            middle_occluded_rows, overlap_points, combined_cluster
        )
        np.testing.assert_array_equal(result, expected_output)

    def test_filter_middle_overlap_combined_cluster_min_cols(self):
        middle_occluded_rows = [0, 1]
        overlap_points = np.array([[0, 2], [1, 2]])
        combined_cluster = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 1], [1, 2], [1, 3]]
        )
        expected_output = np.array([[0, 0], [0, 1], [0, 4], [0, 5]])
        result = filter_middle_overlap_combined_cluster(
            middle_occluded_rows, overlap_points, combined_cluster, min_cols=2
        )
        np.testing.assert_array_equal(result, expected_output)


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
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
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
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
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
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 2.0
        expected_angle_pred = 0.0
        expected_support = 4

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
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 1.5
        expected_angle_pred = 0.0
        expected_support = 2

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
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 1.9545454545454548
        expected_angle_pred = -27.743204472006298
        expected_support = 2
        expected_r = 1.0
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(r, expected_r, places=2)

    def test_calculate_position_from_cluster_and_image_only_top_rows(self):
        img = np.array(
            [
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [1, 2]])
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )
        expected_pos = 4
        expected_angle_pred = -45
        expected_support = 2
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)


@pytest.mark.skip(reason="Tmp disable")
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
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2),
            "middle_occluded",
        )

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


@pytest.mark.skip(reason="Tmp disable")
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


class TestCheckOcclusionTypeOfASingleCluster(unittest.TestCase):
    def test_no_saved_darts(self):
        # Test when there are no saved darts
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        result = check_occlusion_type_of_a_single_cluster(cluster_in, saved_darts={})
        self.assertEqual(result, {})

    def test_no_overlap(self):
        # Test when there's no overlap with saved darts
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[5, 5], [6, 6], [7, 7]])}}
        result = check_occlusion_type_of_a_single_cluster(cluster_in, saved_darts)
        self.assertEqual(result, {})

    def test_single_dart_overlap(self):
        # Test with overlap with one saved dart
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[1, 1], [3, 3], [4, 4]])}}
        result = check_occlusion_type_of_a_single_cluster(cluster_in, saved_darts)

        # Check that result is not empty
        self.assertNotEqual(result, {})

        # Check that the result contains the expected keys
        expected_keys = [
            "overlapping_darts",
            "occlusion_kind",
            "overlap_points",
            "fully_usable_rows",
            "middle_occluded_rows",
            "single_pixel_thick_overlap_rows",
            "left_side_overlap_rows",
            "right_side_overlap_rows",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # Check specific values
        self.assertEqual(result["overlapping_darts"], [1])
        self.assertIn(
            result["occlusion_kind"],
            [
                "fully_useable",
                "middle_occluded",
                "left_side_fully_occluded",
                "right_side_fully_occluded",
            ],
        )
        # Check that overlap_points contains the expected point
        overlap_points_as_tuples = [tuple(point) for point in result["overlap_points"]]
        self.assertIn((1, 1), overlap_points_as_tuples)

    def test_multiple_darts_overlap(self):
        # Test with overlap with multiple saved darts
        cluster_in = np.array([[0, 1], [1, 1], [1, 2], [0, 2]])
        saved_darts = {
            "d1": {"cluster": np.array([[0, 1], [1, 1]])},
            "d2": {"cluster": np.array([[0, 4], [1, 4]])},
        }
        result = check_occlusion_type_of_a_single_cluster(cluster_in, saved_darts)

        # Check that result is not empty
        self.assertNotEqual(result, {})

        # Check that the result contains the expected keys
        expected_keys = [
            "overlapping_darts",
            "occlusion_kind",
            "overlap_points",
            "fully_usable_rows",
            "middle_occluded_rows",
            "single_pixel_thick_overlap_rows",
            "left_side_overlap_rows",
            "right_side_overlap_rows",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # Check specific values
        self.assertEqual(sorted(result["overlapping_darts"]), [1])
        self.assertIn(
            result["occlusion_kind"],
            [
                "fully_useable",
                "middle_occluded",
                "left_side_fully_occluded",
                "right_side_fully_occluded",
            ],
        )

        # Check that overlap_points contains the expected points
        overlap_points_as_tuples = [tuple(point) for point in result["overlap_points"]]
        self.assertIn((1, 1), overlap_points_as_tuples)
        self.assertIn((0, 1), overlap_points_as_tuples)


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

        pos, angle, support, r, error, cluster = calculate_position_from_occluded_dart(
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
            "occlusion_kind": "left_side_fully_occluded",
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

        pos, angle, support, r, error, cluster = calculate_position_from_occluded_dart(
            occlusion_dict, cluster_in, diff_img, current_img, saved_darts
        )

        expected_pos = 0.75
        expected_angle = 0
        expected_support = 3
        expected_error = 0.25

        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle, expected_angle, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(error, expected_error, places=2)


class TestCheckWhichSidesAreOccludedOfTheClusters(unittest.TestCase):
    def test_check_which_sides_are_occluded_of_the_clusters_overlap(self):
        clusters = [
            np.array([[0, 0], [1, 1], [2, 2], [3, 1]]),
            np.array([[3, 3], [4, 4], [5, 5], [6, 4]]),
        ]
        overlap = np.array([[1, 1], [4, 4]])

        expected_which_side_overlap = {0: "fully_useable", 1: "fully_useable"}
        expected_occluded_rows_clusters = {
            0: {
                "fully_usable_rows": [0, 2, 3],
                "middle_occluded_rows": [],
                "left_side_overlap_rows": [],
                "right_side_overlap_rows": [],
                "single_pixel_thick_overlap_rows": [1],
            },
            1: {
                "fully_usable_rows": [3, 5, 6],
                "middle_occluded_rows": [],
                "left_side_overlap_rows": [],
                "right_side_overlap_rows": [],
                "single_pixel_thick_overlap_rows": [4],
            },
        }

        which_side_overlap, occluded_rows_clusters = (
            check_which_sides_are_occluded_of_the_clusters(clusters, overlap)
        )
        self.assertDictEqual(which_side_overlap, expected_which_side_overlap)
        self.assertDictEqual(occluded_rows_clusters, expected_occluded_rows_clusters)
