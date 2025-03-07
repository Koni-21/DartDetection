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
    """Tests for the dilate_cluster function."""

    def test_dilate_cluster_no_dilation(self):
        """Test when dilation amount is 0, cluster should remain unchanged."""
        # Arrange
        cluster_mask = np.array([[0, 1], [1, 1], [2, 1]])
        img_width = 5
        dilate_cluster_by_n_px = 0
        expected_output = np.array([[0, 1], [1, 1], [2, 1]])

        # Act
        result = dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px)

        # Assert
        np.testing.assert_array_equal(result, expected_output)

    def test_dilate_cluster_single_pixel_dilation(self):
        """Test dilation by 1 pixel in each direction."""
        # Arrange
        cluster_mask = np.array([[0, 1], [1, 1], [2, 1]])
        img_width = 5
        dilate_cluster_by_n_px = 1
        expected_output = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        # Act
        result = dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px)

        # Assert
        np.testing.assert_array_equal(result, expected_output)

    def test_dilate_cluster_multiple_pixel_dilation(self):
        """Test dilation by multiple pixels."""
        # Arrange
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

        # Act
        result = dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px)

        # Assert
        np.testing.assert_array_equal(result, expected_output)

    def test_dilate_cluster_boundary_conditions(self):
        """Test dilation when cluster is at the edge of image boundaries."""
        # Arrange
        cluster_mask = np.array([[0, 0], [1, 0], [2, 0]])
        img_width = 3
        dilate_cluster_by_n_px = 1
        expected_output = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])

        # Act
        result = dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px)

        # Assert
        np.testing.assert_array_equal(result, expected_output)

    def test_dilate_cluster_empty(self):
        """Test dilation with an empty cluster."""
        # Arrange
        cluster_mask = np.array([]).reshape(0, 2)
        img_width = 5
        dilate_cluster_by_n_px = 1
        expected_output = np.array([]).reshape(0, 2)

        # Act
        result = dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px)

        # Assert
        np.testing.assert_array_equal(result, expected_output)


class TestOverlap(unittest.TestCase):
    """Tests for the overlap function."""

    def test_overlap_no_overlap(self):
        """Test case when clusters have no overlapping points."""
        # Arrange
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        saved_dart_cluster = np.array([[3, 3], [4, 4], [5, 5]])
        expected_output = np.array([])

        # Act & Assert
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

    def test_overlap_partial_overlap(self):
        """Test case when clusters have some overlapping points."""
        # Arrange
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        saved_dart_cluster = np.array([[1, 1], [3, 3], [4, 4]])
        expected_output = np.array([[1, 1]])

        # Act & Assert
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

    def test_overlap_full_overlap(self):
        """Test case when clusters completely overlap."""
        # Arrange
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        saved_dart_cluster = np.array([[0, 0], [1, 1], [2, 2]])
        expected_output = np.array([[0, 0], [1, 1], [2, 2]])

        # Act & Assert
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

    def test_overlap_empty_clusters(self):
        """Test case with empty clusters."""
        # Arrange
        cluster = np.array([])
        saved_dart_cluster = np.array([])
        expected_output = np.array([])

        # Act & Assert
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

    def test_overlap_one_empty_cluster(self):
        """Test cases where one cluster is empty."""
        # Arrange - First test
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        saved_dart_cluster = np.array([])
        expected_output = np.array([])

        # Act & Assert - First test
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )

        # Arrange - Second test
        cluster = np.array([])
        saved_dart_cluster = np.array([[0, 0], [1, 1], [2, 2]])

        # Act & Assert - Second test
        np.testing.assert_array_equal(
            overlap(cluster, saved_dart_cluster), expected_output
        )


class TestDifferentiateOverlap(unittest.TestCase):
    """Tests for the differentiate_overlap function."""

    def test_differentiate_overlap_partial_overlap(self):
        """Test case with partial overlap."""
        # Arrange
        cluster = np.array([[0, 0], [1, 1], [2, 2]])
        overlap_points = np.array([[1, 1]])
        expected_output = {
            "fully_usable_rows": [0, 2],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [1],
        }

        # Act & Assert
        self.assertDictEqual(
            differentiate_overlap(cluster, overlap_points), expected_output
        )

    def test_differentiate_overlap_left_and_right_overlap(self):
        """Test case with left and right side overlaps."""
        # Arrange
        cluster = np.array([[0, 0], [1, 1], [2, 2], [1, 0], [2, 1]])
        overlap_points = np.array([[0, 0], [1, 1], [2, 2]])
        expected_output = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [1, 2],
            "single_pixel_thick_overlap_rows": [0],
        }

        # Act & Assert
        self.assertDictEqual(
            differentiate_overlap(cluster, overlap_points), expected_output
        )

    def test_differentiate_overlap_middle_overlap(self):
        """Test case with middle overlap."""
        # Arrange
        cluster = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 23]])
        overlap_points = np.array([[0, 2], [1, 2]])
        expected_output = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [0, 1],
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act & Assert
        self.assertDictEqual(
            differentiate_overlap(cluster, overlap_points), expected_output
        )

    def test_differentiate_overlap_left_side(self):
        """Test case with left side overlap."""
        # Arrange
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2]])
        overlap_points = np.array([[0, 1], [1, 1]])
        expected_output = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [0, 1],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act & Assert
        self.assertDictEqual(
            differentiate_overlap(cluster, overlap_points), expected_output
        )

    def test_differentiate_overlap_right_side(self):
        """Test case with right side overlap."""
        # Arrange
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2]])
        overlap_points = np.array([[0, 2], [1, 2]])
        expected_output = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [0, 1],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act & Assert
        self.assertDictEqual(
            differentiate_overlap(cluster, overlap_points), expected_output
        )


class TestFilterClusterByUsableRows(unittest.TestCase):
    """Tests for filter_cluster_by_usable_rows function."""

    def test_filter_cluster_by_usable_rows_all_usable(self):
        """Test when all rows are marked as usable."""
        # Arrange
        cluster = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
        usable_rows = [0, 1]
        expected_output = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])

        # Act & Assert
        np.testing.assert_array_equal(
            filter_cluster_by_usable_rows(usable_rows, cluster), expected_output
        )

    def test_filter_cluster_by_usable_rows_some_usable(self):
        """Test when only some rows are marked as usable."""
        # Arrange
        cluster = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]]
        )
        usable_rows = [0, 1]
        expected_output = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])

        # Act & Assert
        np.testing.assert_array_equal(
            filter_cluster_by_usable_rows(usable_rows, cluster), expected_output
        )

    def test_filter_cluster_by_usable_rows_empty(self):
        """Test when usable rows list is empty."""
        # Arrange
        cluster = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
        usable_rows = []
        expected_output = np.array([]).reshape(0, 2)

        # Act & Assert
        result = filter_cluster_by_usable_rows(usable_rows, cluster)
        self.assertEqual(result.shape[0], 0)
        self.assertEqual(result.shape[1], 2)


class TestFilterMiddleOverlapCombinedCluster(unittest.TestCase):
    """Tests for filter_middle_overlap_combined_cluster function."""

    def test_filter_middle_overlap_combined_cluster_standard(self):
        """Test standard case for middle overlap filtering."""
        # Arrange
        middle_occluded_rows = [0, 1]
        overlap_points = np.array([[0, 2], [1, 2]])
        combined_cluster = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
        expected_output = np.array([[0, 1], [0, 3], [1, 1], [1, 3]])

        # Act
        result = filter_middle_overlap_combined_cluster(
            middle_occluded_rows, overlap_points, combined_cluster
        )

        # Assert
        np.testing.assert_array_equal(result, expected_output)

    def test_filter_middle_overlap_combined_cluster_cutoff_not_symetric_part(self):
        """Test filtering when the overlap is not symmetrical."""
        # Arrange
        middle_occluded_rows = [0, 1]
        overlap_points = np.array([[0, 2], [1, 2]])
        combined_cluster = np.array(
            [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 1], [1, 2], [1, 3]]
        )
        expected_output = np.array([[0, 1], [0, 5], [1, 1], [1, 3]])

        # Act
        result = filter_middle_overlap_combined_cluster(
            middle_occluded_rows, overlap_points, combined_cluster
        )

        # Assert
        np.testing.assert_array_equal(result, expected_output)

    def test_filter_middle_overlap_combined_cluster_min_cols(self):
        """Test filtering with minimum column constraint."""
        # Arrange
        middle_occluded_rows = [0, 1]
        overlap_points = np.array([[0, 2], [1, 2]])
        combined_cluster = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 1], [1, 2], [1, 3]]
        )
        expected_output = np.array([[0, 0], [0, 1], [0, 4], [0, 5]])

        # Act
        result = filter_middle_overlap_combined_cluster(
            middle_occluded_rows, overlap_points, combined_cluster, min_cols=2
        )

        # Assert
        np.testing.assert_array_equal(result, expected_output)

    def test_filter_middle_overlap_combined_cluster_empty(self):
        """Test filtering with empty inputs."""
        # Arrange
        middle_occluded_rows = []
        overlap_points = np.array([]).reshape(0, 2)
        combined_cluster = np.array([[0, 1], [0, 2], [0, 3]])

        # Act
        result = filter_middle_overlap_combined_cluster(
            middle_occluded_rows, overlap_points, combined_cluster
        )

        # Assert
        np.testing.assert_array_equal(result, np.array([]).reshape(0, 2))


class TestCalculatePositionFromClusterAndImage(unittest.TestCase):
    """Tests for calculate_position_from_cluster_and_image function."""

    def test_calculate_position_from_cluster_and_image_simple(self):
        """Test simple vertical dart position calculation."""
        # Arrange
        img = np.array(
            [
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])

        # Act
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )

        # Assert
        expected_pos = 1
        expected_angle_pred = 0
        expected_support = 4
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)

    def test_calculate_position_from_cluster_and_image_diagonal(self):
        """Test diagonal dart position calculation."""
        # Arrange
        img = np.array(
            [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
            ]
        )
        cluster = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

        # Act
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )

        # Assert
        expected_pos = 3.0
        expected_angle_pred = -45.0
        expected_support = 4
        expected_r = 1.0
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(r, expected_r, places=2)

    def test_calculate_position_from_cluster_and_image_subpixel(self):
        """Test subpixel dart position calculation."""
        # Arrange
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

        # Act
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )

        # Assert
        expected_pos = 2.0
        expected_angle_pred = 0.0
        expected_support = 4
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)

    def test_calculate_position_from_cluster_and_image_subpixel_2(self):
        """Test subpixel dart position calculation with different image."""
        # Arrange
        img = np.array(
            [
                [1, 0.4, 0.4, 1, 1],
                [1, 0.5, 0.5, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2]])

        # Act
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )

        # Assert
        expected_pos = 1.5
        expected_angle_pred = 0.0
        expected_support = 2
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)

    def test_calculate_position_from_cluster_and_image_subpixel_3(self):
        """Test subpixel dart position calculation with another image."""
        # Arrange
        img = np.array(
            [
                [1, 0.2, 0.4, 1, 1],
                [1, 0.1, 0.5, 0.2, 1],
            ]
        )
        cluster = np.array([[0, 1], [0, 2], [1, 1], [1, 2], [1, 3]])

        # Act
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )

        # Assert
        expected_pos = 1.9545454545454548
        expected_angle_pred = -27.743204472006298
        expected_support = 2
        expected_r = 1.0
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(r, expected_r, places=2)

    def test_calculate_position_from_cluster_and_image_only_top_rows(self):
        """Test dart position calculation with only top rows."""
        # Arrange
        img = np.array(
            [
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        cluster = np.array([[0, 1], [1, 2]])

        # Act
        pos, angle_pred, support, r, error = calculate_position_from_cluster_and_image(
            img, cluster
        )

        # Assert
        expected_pos = 4
        expected_angle_pred = -45
        expected_support = 2
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle_pred, expected_angle_pred, places=2)
        self.assertEqual(support, expected_support)


class TestOcclusionKind(unittest.TestCase):
    """Tests for occlusion_kind function."""

    def test_occlusion_kind_fully_usable(self):
        """Test when there are enough fully usable rows."""
        # Arrange
        occluded_rows = {
            "fully_usable_rows": [0, 1, 2],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act & Assert
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2), "fully_useable"
        )

    def test_occlusion_kind_middle_occluded(self):
        """Test when middle part is occluded."""
        # Arrange
        occluded_rows = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [0, 1, 2],
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act & Assert
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2),
            "middle_occluded",
        )

    def test_occlusion_kind_left_side_fully_occluded(self):
        """Test when left side is fully occluded."""
        # Arrange
        occluded_rows = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [0, 1, 2],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act & Assert
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2),
            "left_side_fully_occluded",
        )

    def test_occlusion_kind_right_side_fully_occluded(self):
        """Test when right side is fully occluded."""
        # Arrange
        occluded_rows = {
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [0, 1, 2],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act & Assert
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2),
            "right_side_fully_occluded",
        )

    def test_occlusion_kind_not_enough_usable_rows(self):
        """Test when there aren't enough usable rows."""
        # Arrange
        occluded_rows = {
            "fully_usable_rows": [0],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act & Assert
        self.assertEqual(
            occlusion_kind(occluded_rows, thresh_needed_rows=2),
            "undefined_overlap_case",
        )


@pytest.mark.skip(reason="Tmp disable")
class TestCheckOverlap(unittest.TestCase):
    """Tests for check_overlap function."""

    def test_check_overlap_no_saved_darts(self):
        """Test when there are no saved darts."""
        # Arrange
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {}
        expected_output = {}

        # Act
        result = check_overlap(cluster_in, saved_darts)

        # Assert
        self.assertDictEqual(result, expected_output)

    def test_check_overlap_no_overlap(self):
        """Test when there's no overlap with saved darts."""
        # Arrange
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[3, 3], [4, 4], [5, 5]])}}
        expected_output = {}

        # Act & Assert
        self.assertDictEqual(check_overlap(cluster_in, saved_darts), expected_output)

    def test_check_overlap_partial_overlap(self):
        """Test with partial overlap with saved darts."""
        # Arrange
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[1, 1], [3, 3], [4, 4]])}}
        expected_output = {
            "overlapping_darts": [1],
            "occlusion_kind": "one_side_fully_occluded",
            "overlap_points": np.array([[1, 1]]),
            "fully_usable_rows": [0, 2],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [1],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act
        result = check_overlap(cluster_in, saved_darts)

        # Assert
        for key in expected_output:
            if isinstance(expected_output[key], np.ndarray):
                np.testing.assert_array_equal(result[key], expected_output[key])
            else:
                self.assertEqual(result[key], expected_output[key])

    def test_check_overlap_full_overlap(self):
        """Test with full overlap with saved darts."""
        # Arrange
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[0, 0], [1, 1], [2, 2]])}}
        expected_output = {
            "overlapping_darts": [1],
            "occlusion_kind": "one_side_fully_occluded",
            "overlap_points": np.array([[0, 0], [1, 1], [2, 2]]),
            "fully_usable_rows": [],
            "middle_occluded_rows": [],
            "left_side_overlap_rows": [0, 1, 2],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act
        result = check_overlap(cluster_in, saved_darts)

        # Assert
        for key in expected_output:
            if isinstance(expected_output[key], np.ndarray):
                np.testing.assert_array_equal(result[key], expected_output[key])
            else:
                self.assertEqual(result[key], expected_output[key])

    def test_check_overlap_multiple_overlaps(self):
        """Test with multiple overlaps with saved darts."""
        # Arrange
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
            "left_side_overlap_rows": [0, 1, 2],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act
        result = check_overlap(cluster_in, saved_darts)

        # Assert
        for key in expected_output:
            if isinstance(expected_output[key], np.ndarray):
                np.testing.assert_array_equal(result[key], expected_output[key])
            else:
                self.assertEqual(result[key], expected_output[key])

    def test_check_overlap_middle_occlusion(self):
        """Test with middle occlusion."""
        # Arrange
        cluster_in = np.array([[0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]])
        saved_darts = {"d1": {"cluster": np.array([[0, 2], [1, 2]])}}
        expected_output = {
            "overlapping_darts": [1],
            "occlusion_kind": "middle_occluded",
            "overlap_points": np.array([[0, 2], [1, 2]]),
            "fully_usable_rows": [],
            "middle_occluded_rows": [0, 1],
            "left_side_overlap_rows": [],
            "right_side_overlap_rows": [],
            "single_pixel_thick_overlap_rows": [],
        }

        # Act
        result = check_overlap(cluster_in, saved_darts)

        # Assert
        for key in expected_output:
            if isinstance(expected_output[key], np.ndarray):
                np.testing.assert_array_equal(result[key], expected_output[key])
            else:
                self.assertEqual(result[key], expected_output[key])


class TestCheckOcclusionTypeOfASingleCluster(unittest.TestCase):
    """Tests for check_occlusion_type_of_a_single_cluster function."""

    def test_no_saved_darts(self):
        """Test when there are no saved darts."""
        # Arrange
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])

        # Act
        result = check_occlusion_type_of_a_single_cluster(cluster_in, saved_darts={})

        # Assert
        self.assertEqual(result, {})

    def test_no_overlap(self):
        """Test when there's no overlap with saved darts."""
        # Arrange
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[5, 5], [6, 6], [7, 7]])}}

        # Act
        result = check_occlusion_type_of_a_single_cluster(cluster_in, saved_darts)

        # Assert
        self.assertEqual(result, {})

    def test_single_dart_overlap(self):
        """Test with overlap with one saved dart."""
        # Arrange
        cluster_in = np.array([[0, 0], [1, 1], [2, 2]])
        saved_darts = {"d1": {"cluster": np.array([[1, 1], [3, 3], [4, 4]])}}

        # Act
        result = check_occlusion_type_of_a_single_cluster(cluster_in, saved_darts)

        # Assert
        self.assertNotEqual(result, {})
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
        overlap_points_as_tuples = [tuple(point) for point in result["overlap_points"]]
        self.assertIn((1, 1), overlap_points_as_tuples)

    def test_multiple_darts_overlap(self):
        """Test with overlap with multiple saved darts."""
        # Arrange
        cluster_in = np.array([[0, 1], [1, 1], [1, 2], [0, 2]])
        saved_darts = {
            "d1": {"cluster": np.array([[0, 1], [1, 1]])},
            "d2": {"cluster": np.array([[0, 4], [1, 4]])},
        }

        # Act
        result = check_occlusion_type_of_a_single_cluster(cluster_in, saved_darts)

        # Assert
        self.assertNotEqual(result, {})
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
        overlap_points_as_tuples = [tuple(point) for point in result["overlap_points"]]
        self.assertIn((1, 1), overlap_points_as_tuples)
        self.assertIn((0, 1), overlap_points_as_tuples)


class TestCalculatePositionFromOccludedDart(unittest.TestCase):
    """Tests for calculate_position_from_occluded_dart function."""

    def test_calculate_position_from_occluded_dart_fully_usable(self):
        """Test when occlusion kind is fully usable."""
        # Arrange
        occlusion_dict = {
            "occlusion_kind": "fully_useable",
            "fully_usable_rows": [0, 1, 2],
        }
        cluster_in = np.array([[0, 1], [1, 1], [2, 1]])
        diff_img = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        current_img = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        saved_darts = {}

        # Act
        pos, angle, support, r, error, cluster = calculate_position_from_occluded_dart(
            occlusion_dict, cluster_in, diff_img, current_img, saved_darts
        )

        # Assert
        expected_pos = 1
        expected_angle = 0
        expected_support = 3
        expected_error = 0
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle, expected_angle, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(error, expected_error, places=2)

    def test_calculate_position_from_occluded_dart_one_side_fully_occluded(self):
        """Test when occlusion kind is one side fully occluded."""
        # Arrange
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

        # Act
        pos, angle, support, r, error, cluster = calculate_position_from_occluded_dart(
            occlusion_dict, cluster_in, diff_img, current_img, saved_darts
        )

        # Assert
        expected_pos = 0.75
        expected_angle = 0
        expected_support = 3
        expected_error = 0.25
        self.assertAlmostEqual(pos, expected_pos, places=2)
        self.assertAlmostEqual(angle, expected_angle, places=2)
        self.assertEqual(support, expected_support)
        self.assertAlmostEqual(error, expected_error, places=2)


class TestCheckWhichSidesAreOccludedOfTheClusters(unittest.TestCase):
    """Tests for check_which_sides_are_occluded_of_the_clusters function."""

    def test_check_which_sides_are_occluded_of_the_clusters_overlap(self):
        """Test overlap detection with two clusters."""
        # Arrange
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

        # Act
        which_side_overlap, occluded_rows_clusters = (
            check_which_sides_are_occluded_of_the_clusters(clusters, overlap)
        )

        # Assert
        self.assertDictEqual(which_side_overlap, expected_which_side_overlap)
        self.assertDictEqual(occluded_rows_clusters, expected_occluded_rows_clusters)

    def test_check_which_sides_are_occluded_of_the_clusters_no_overlap(self):
        """Test when there's no overlap between clusters."""
        # Arrange
        clusters = [
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[3, 3], [4, 4], [5, 5]]),
        ]
        overlap = np.array([]).reshape(0, 2)

        expected_which_side_overlap = {0: "fully_useable", 1: "fully_useable"}
        expected_occluded_rows_clusters = {
            0: {
                "fully_usable_rows": [0, 1, 2],
                "middle_occluded_rows": [],
                "left_side_overlap_rows": [],
                "right_side_overlap_rows": [],
                "single_pixel_thick_overlap_rows": [],
            },
            1: {
                "fully_usable_rows": [3, 4, 5],
                "middle_occluded_rows": [],
                "left_side_overlap_rows": [],
                "right_side_overlap_rows": [],
                "single_pixel_thick_overlap_rows": [],
            },
        }

        # Act
        which_side_overlap, occluded_rows_clusters = (
            check_which_sides_are_occluded_of_the_clusters(clusters, overlap)
        )

        # Assert
        self.assertDictEqual(which_side_overlap, expected_which_side_overlap)
        self.assertDictEqual(occluded_rows_clusters, expected_occluded_rows_clusters)
