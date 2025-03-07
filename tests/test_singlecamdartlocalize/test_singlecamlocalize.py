import os
import pathlib
import logging
import unittest
import pytest

import numpy as np
from matplotlib import pyplot as plt

from dartdetect.singlecamdartlocalize.singlecamlocalize import (
    SingleCamLocalize,
)

from dartdetect.singlecamdartlocalize.simulationtestutils import draw_dart_subpixel

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


class TestSingleCamLocalize(unittest.TestCase):
    def setUp(self):
        """Initialize a fresh SingleCamLocalize object before each test."""
        self.Loc = SingleCamLocalize()

    def test_new_image_single_image(self):
        """Test that adding a single image stores it correctly but returns None."""
        # Arrange
        img = np.ones((5, 5))

        # Act
        result = self.Loc.new_image(img)

        # Assert
        self.assertIsNone(result, "First image should return None")
        self.assertEqual(self.Loc.image_count, 1, "Image count should be incremented")
        self.assertEqual(len(self.Loc.imgs), 1, "Image should be stored in imgs list")
        np.testing.assert_array_equal(
            self.Loc.current_img, img, "Current image should match input"
        )

    def test_new_image_multiple_images(self):
        """Test that adding multiple images triggers analysis."""
        # Arrange
        img1 = np.ones((5, 5))
        img2 = np.ones((5, 5)) * 0.5

        # Act
        self.Loc.new_image(img1)
        result = self.Loc.new_image(img2)

        # Assert
        self.assertIsNotNone(result, "Second image should return analysis result")
        self.assertEqual(
            self.Loc.image_count, 2, "Image count should be incremented to 2"
        )
        self.assertEqual(len(self.Loc.imgs), 2, "Two images should be stored")
        np.testing.assert_array_equal(
            self.Loc.current_img, img2, "Current image should be updated"
        )

    def test_analyse_imgs_no_change(self):
        """Test that identical images produce no detection result."""
        # Arrange
        img1 = np.ones((5, 5))
        img2 = np.ones((5, 5))

        # Act
        self.Loc.new_image(img1)
        result = self.Loc.new_image(img2)

        # Assert
        self.assertIsNone(result, "Identical images should not detect changes")

    def test_incoming_cluster_detected(self):
        """Test detection of a new dart with a simple line pattern."""
        # Arrange
        img1 = np.ones((3, 5))
        img2 = np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
            ]
        )

        # Act
        self.Loc.new_image(img1)
        result = self.Loc.new_image(img2)

        # Assert
        self.assertIsNotNone(result, "Dart should be detected")
        self.assertIn("pos", result, "Position should be in result")
        self.assertEqual(2, result["pos"], "Position should match expected column")
        self.assertIn("angle", result, "Angle should be in result")
        self.assertIn("r", result, "Radius should be in result")
        self.assertIn("support", result, "Support should be in result")
        self.assertIn("error", result, "Error should be in result")

    def test_leaving_cluster_not_arrived_detected(self):
        """Test that dart removal is handled correctly when dart hasn't fully arrived."""
        # Arrange
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

        # Act
        self.Loc.new_image(img1)
        result = self.Loc.new_image(img2)

        # Assert
        self.assertIsNone(result, "No result expected for non-fully arrived dart")
        self.assertEqual(len(self.Loc.saved_darts), 0, "No darts should be saved")

    def test_incoming_and_leaving_cluster_detected(self):
        """Test that dart arrival and removal sequence works correctly."""
        # Arrange
        img1 = np.ones((3, 5))
        img2 = np.array(
            [
                [1, 1, 0.4, 0, 1],
                [1, 1, 0.5, 0.1, 1],
                [1, 1, 0.2, 0.4, 1],
            ]
        )
        img3 = np.ones((3, 5))

        # Act & Assert - Arrival
        self.Loc.new_image(img1)
        result_incoming = self.Loc.new_image(img2)
        self.assertIsNotNone(result_incoming, "Dart arrival should be detected")
        self.assertEqual(len(self.Loc.saved_darts), 1, "One dart should be saved")

        # Act & Assert - Removal
        result_leaving = self.Loc.new_image(img3)
        self.assertIsNone(result_leaving, "No new dart detected during removal")
        self.assertEqual(len(self.Loc.saved_darts), 0, "All darts should be removed")

    def test_visualize_stream(self):
        """Test that visualization works correctly with detected darts."""
        # Arrange
        img1 = np.ones((3, 5))
        img2 = np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
            ]
        )
        fig, ax = plt.subplots()

        # Act
        self.Loc.new_image(img1)
        self.Loc.new_image(img2)
        self.Loc.visualize_stream(ax)

        # Assert
        self.assertEqual(len(ax.lines), 1, "One line should be drawn for dart angle")

    def test_only_part_occluded_fully_usable_rows(self):
        """Test detection with partial occlusion but usable rows."""
        # Arrange
        img0 = np.ones([20, 500])
        img_d1 = draw_dart_subpixel(img0.copy(), 200, 0, 10)
        img_d2 = draw_dart_subpixel(img_d1.copy(), 215, 20, 15)
        self.Loc.thresh_binarise_cluster = 0
        self.Loc.thresh_noise = 0

        # Act
        self.Loc.new_image(img0)
        d1 = self.Loc.new_image(img_d1)
        d2 = self.Loc.new_image(img_d2)

        # Assert
        self.assertAlmostEqual(
            d1["pos"], 200, msg="First dart position should be at x=200"
        )
        self.assertAlmostEqual(
            d2["pos"],
            215.04274345247006,
            msg="Second dart position should be correctly detected",
        )
        self.assertAlmostEqual(
            d2["angle"],
            19.797940072766973,
            msg="Second dart angle should be correctly detected",
        )

    def test_one_side_fully_occluded(self):
        """Test detection with one side fully occluded."""
        # Arrange
        img0 = np.ones([20, 500])
        img_d1 = draw_dart_subpixel(img0.copy(), 200, 0, 10)
        img_d2 = draw_dart_subpixel(img_d1.copy(), 210, 5, 15)
        self.Loc.thresh_binarise_cluster = 0
        self.Loc.thresh_noise = 0

        # Act
        self.Loc.new_image(img0)
        d1 = self.Loc.new_image(img_d1)
        d2 = self.Loc.new_image(img_d2)

        # Assert
        self.assertAlmostEqual(
            d1["pos"], 200, msg="First dart position should be at x=200"
        )
        self.assertAlmostEqual(
            d2["pos"],
            209.07443785489002,
            msg="Second dart with occlusion should be correctly detected",
        )
        self.assertAlmostEqual(
            d2["angle"],
            2.631871731830313,
            msg="Second dart angle with occlusion should be correct",
        )
        self.assertAlmostEqual(
            d2["error"], 2.750036570800802, msg="Error metric should reflect occlusion"
        )

    def test_only_middle_occluded(self):
        """Test detection with only the middle occluded."""
        # Arrange
        img0 = np.ones([20, 500])
        img_d1 = draw_dart_subpixel(img0.copy(), 200, 0, 10)
        img_d2 = draw_dart_subpixel(img_d1.copy(), 203.1, 0, 25)
        self.Loc.thresh_binarise_cluster = 0
        self.Loc.thresh_noise = 0
        self.Loc.dilate_cluster_by_n_px = 1
        self.Loc.min_usable_columns_middle_overlap = 1

        # Act
        self.Loc.new_image(img0)
        d1 = self.Loc.new_image(img_d1)
        _ = self.Loc.new_image(
            img_d2
        )  # When there are multiple clusters wait two frames for stability
        d2 = self.Loc.new_image(img_d2)

        # Assert
        self.assertAlmostEqual(
            d1["pos"], 200, msg="First dart position should be at x=200"
        )
        self.assertListEqual(
            [d2["pos"], d2["angle"], d2["error"], d2["support"]],
            [203.0, -0.0, 0.22360679774997896, 20],
            msg="Second dart with middle occlusion should have expected values",
        )

    def test_reset(self):
        """Test that the reset method properly clears all state."""
        # Arrange
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

        # Act
        self.Loc.reset()

        # Assert
        self.assertEqual(self.Loc.image_count, 0, "Image count should be reset to 0")
        self.assertEqual(len(self.Loc.imgs), 0, "Image list should be empty")
        self.assertIsNone(self.Loc.current_img, "Current image should be None")
        self.assertEqual(len(self.Loc.saved_darts), 0, "Saved darts should be empty")
        self.assertEqual(self.Loc.dart, 0, "Dart count should be reset")

    def test_check_view_empty(self):
        """Test that the check view empty method properly clears all state."""
        # Arrange
        img1 = np.ones((3, 5))
        img2 = np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1],
            ]
        )
        img3 = np.ones((3, 5))

        self.Loc.new_image(img1)
        self.Loc.new_image(img2)

        # Act
        self.Loc.new_image(img3)

        # Assert
        self.assertEqual(self.Loc.image_count, 0, "Image count should be reset to 0")
        self.assertEqual(len(self.Loc.imgs), 0, "Image list should be empty")
        self.assertIsNone(self.Loc.current_img, "Current image should be None")
        self.assertEqual(len(self.Loc.saved_darts), 0, "Saved darts should be empty")
        self.assertEqual(self.Loc.dart, 0, "Dart count should be reset")


class TestSingleCamLocalize_real_world_data(unittest.TestCase):
    """Tests using real-world data from test image files."""

    @classmethod
    def setUpClass(cls):
        """Set up paths for test data."""
        # Define base path relative to the repository root
        cls.base_path = pathlib.Path(__file__).parent.parent.parent
        cls.test_data_path = (
            cls.base_path / "data" / "test_imgs" / "250106_usbcam_imgs" / "imgs_0_to_18"
        )

    def test_images_usb_cam_left(self):
        """Test dart detection using real images from left camera."""
        # Arrange
        folder_path = self.test_data_path / "cam_left"

        # Skip test if the folder doesn't exist
        if not folder_path.exists():
            self.skipTest(f"Test data not found at {folder_path}")

        image_files = sorted([f for f in os.listdir(folder_path)])
        images = [plt.imread(os.path.join(folder_path, f)) for f in image_files]

        # Act
        Loc = SingleCamLocalize()
        for img in images:
            Loc.new_image(img)

        # Assert
        self.assertEqual(len(Loc.saved_darts), 6, "Should detect 6 darts")
        self.assertAlmostEqual(
            Loc.saved_darts["d1"]["pos"], 486.85104802368437, msg="Dart 1 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d2"]["pos"], 656.658273636061, msg="Dart 2 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d3"]["pos"], 247.517908029376, msg="Dart 3 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d4"]["pos"], 484.52184735096125, msg="Dart 4 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d5"]["pos"], 606.6639115874592, msg="Dart 5 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d6"]["pos"], 580.896894915437, msg="Dart 6 position"
        )

    def test_images_usb_cam_right(self):
        """Test dart detection using real images from right camera."""
        # Arrange
        folder_path = self.test_data_path / "cam_right"

        # Skip test if the folder doesn't exist
        if not folder_path.exists():
            self.skipTest(f"Test data not found at {folder_path}")

        image_files = sorted([f for f in os.listdir(folder_path)])
        images = [plt.imread(os.path.join(folder_path, f)) for f in image_files]

        # Act
        Loc = SingleCamLocalize()
        for img in images:
            Loc.new_image(img)

        # Assert
        self.assertEqual(len(Loc.saved_darts), 6, "Should detect 6 darts")
        self.assertAlmostEqual(
            Loc.saved_darts["d1"]["pos"], 961.1843337382788, msg="Dart 1 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d2"]["pos"], 536.1432648703126, msg="Dart 2 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d3"]["pos"], 650.3970681707274, msg="Dart 3 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d4"]["pos"], 507.66883553411407, msg="Dart 4 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d5"]["pos"], 704.5066846428723, msg="Dart 5 position"
        )
        self.assertAlmostEqual(
            Loc.saved_darts["d6"]["pos"], 952.0735731554055, msg="Dart 6 position"
        )

    def test_sequential_processing(self):
        """Test processing multiple image sets in sequence."""
        # Arrange
        img_blank = np.ones([20, 500])
        img_set1 = [
            img_blank.copy(),
            draw_dart_subpixel(img_blank.copy(), 100, 0, 10),
            draw_dart_subpixel(img_blank.copy(), 200, 15, 12),
        ]
        img_set2 = [
            img_blank.copy(),
            draw_dart_subpixel(img_blank.copy(), 300, 30, 15),
        ]

        # Act
        Loc = SingleCamLocalize()
        Loc.thresh_binarise_cluster = 0
        Loc.thresh_noise = 0

        # Process first set
        results1 = []
        for img in img_set1:
            result = Loc.new_image(img)
            if result is not None:
                results1.append(result)

        # Reset and process second set
        Loc.reset()
        results2 = []
        for img in img_set2:
            result = Loc.new_image(img)
            if result is not None:
                results2.append(result)

        # Assert
        self.assertEqual(len(results1), 2, "First set should detect 2 darts")
        self.assertEqual(len(results2), 1, "Second set should detect 1 dart")
        self.assertAlmostEqual(
            results1[0]["pos"], 100, msg="First set - first dart position"
        )
        self.assertAlmostEqual(
            results1[1]["pos"],
            200.01731731340968,
            msg="First set - second dart position",
        )
        self.assertAlmostEqual(
            results2[0]["pos"], 299.9875825946005, msg="Second set - dart position"
        )


@pytest.mark.parametrize(
    "position,angle,width,expected_pos,expected_angle",
    [
        (150, 0, 10, 150, 0),
        (200, 15, 12, 200, 15),
        (300, -10, 15, 300, -10),
    ],
)
def test_parameterized_dart_detection(
    position, angle, width, expected_pos, expected_angle
):
    """Parameterized test for dart detection with various positions and angles."""
    # Arrange
    img_blank = np.ones([20, 500])
    img_dart = draw_dart_subpixel(img_blank.copy(), position, angle, width)

    # Act
    Loc = SingleCamLocalize()
    Loc.thresh_binarise_cluster = 0
    Loc.thresh_noise = 0
    Loc.new_image(img_blank)
    result = Loc.new_image(img_dart)

    # Assert
    assert result is not None, "Should detect a dart"
    assert (
        abs(result["pos"] - expected_pos) < 1
    ), f"Position should be close to {expected_pos}"
    assert (
        abs(result["angle"] - expected_angle) < 1
    ), f"Angle should be close to {expected_angle}"


@pytest.mark.parametrize(
    "position1,angle1,expected_pos1,"
    "position2,angle2,expected_pos2,"
    "position3,angle3,expected_pos3",
    [
        (150, 0, 150, 200, -10, 200, 250, 15, 250),
        (200, 0, 200, 202, 0, 205.2, 195, 0, 193.5),
        (310, -15, 310, 280, 15, 280, 300, 0, 300.8),
    ],
)
def test_different_simualted_sequences(
    position1,
    angle1,
    expected_pos1,
    position2,
    angle2,
    expected_pos2,
    position3,
    angle3,
    expected_pos3,
):
    """Parameterized test for dart detection with various positions and angles."""
    # Arrange
    width = 15
    img_blank = np.ones([20, 500])
    img_dart1 = draw_dart_subpixel(img_blank.copy(), position1, angle1, width)
    img_dart2 = draw_dart_subpixel(img_dart1.copy(), position2, angle2, width)
    img_dart3 = draw_dart_subpixel(img_dart2.copy(), position3, angle3, width)

    # Act
    Loc = SingleCamLocalize()
    Loc.thresh_binarise_cluster = 0
    Loc.thresh_noise = 0
    Loc.new_image(img_blank)
    result1 = Loc.new_image(img_dart1)
    result2 = Loc.new_image(img_dart2)
    result3 = Loc.new_image(img_dart3)
    # Assert
    np.testing.assert_almost_equal(
        result1["pos"], expected_pos1, 1
    ), f"Position should be close to {expected_pos1}"
    np.testing.assert_almost_equal(
        result2["pos"], expected_pos2, 1
    ), f"Position should be close to {expected_pos2}"
    np.testing.assert_almost_equal(
        result3["pos"], expected_pos3, 1
    ), f"Position should be close to {expected_pos3}"


if __name__ == "__main__":
    unittest.main()
