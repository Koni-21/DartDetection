import os

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

    def test_only_part_occluded_fully_usable_rows(self):
        img0 = np.ones([20, 500])
        img_d1 = draw_dart_subpixel(img0.copy(), 200, 0, 10)
        img_d2 = draw_dart_subpixel(img_d1.copy(), 215, 20, 15)

        self.Loc.thresh_binarise_cluster = 0
        self.Loc.thresh_noise = 0

        self.Loc.new_image(img0)
        d1 = self.Loc.new_image(img_d1)
        d2 = self.Loc.new_image(img_d2)

        self.assertAlmostEqual(d1["pos"], 200)
        self.assertAlmostEqual(d2["pos"], 215.04274345247006)
        self.assertAlmostEqual(d2["angle"], 19.797940072766973)

    def test_one_side_fully_occluded(self):
        img0 = np.ones([20, 500])
        img_d1 = draw_dart_subpixel(img0.copy(), 200, 0, 10)
        img_d2 = draw_dart_subpixel(img_d1.copy(), 210, 5, 15)

        self.Loc.thresh_binarise_cluster = 0
        self.Loc.thresh_noise = 0

        self.Loc.new_image(img0)
        d1 = self.Loc.new_image(img_d1)
        d2 = self.Loc.new_image(img_d2)

        self.assertAlmostEqual(d1["pos"], 200)
        self.assertAlmostEqual(d2["pos"], 209.07443785489002)
        self.assertAlmostEqual(d2["angle"], 2.631871731830313)
        self.assertAlmostEqual(d2["error"], 2.750036570800802)

    def test_only_middle_occluded(self):
        img0 = np.ones([20, 500])
        img_d1 = draw_dart_subpixel(img0.copy(), 200, 0, 10)
        img_d2 = draw_dart_subpixel(img_d1.copy(), 203.1, 0, 25)

        self.Loc.thresh_binarise_cluster = 0
        self.Loc.thresh_noise = 0
        self.Loc.dilate_cluster_by_n_px = 1
        self.Loc.min_usable_columns_middle_overlap = 1

        self.Loc.new_image(img0)
        d1 = self.Loc.new_image(img_d1)
        _ = self.Loc.new_image(
            img_d2
        )  # When there are multiple clusters wait two frames for stability
        d2 = self.Loc.new_image(img_d2)

        self.assertAlmostEqual(d1["pos"], 200)
        self.assertListEqual(
            [d2["pos"], d2["angle"], d2["error"], d2["support"]],
            [203.0, -0.0, 0.22360679774997896, 20],
        )


class TestSingleCamLocalize_real_world_data(unittest.TestCase):
    def test_images_usb_cam_left(self):
        folder_path = (
            r"..\DartDetection\data\test_imgs\250106_usbcam_imgs\imgs_0_to_18\cam_left"
        )

        image_files = sorted([f for f in os.listdir(folder_path)])
        images = [plt.imread(os.path.join(folder_path, f)) for f in image_files]

        Loc = SingleCamLocalize()

        for img in images:
            Loc.new_image(img)

        self.assertAlmostEqual(Loc.saved_darts["d1"]["pos"], 486.85104802368437)
        self.assertAlmostEqual(Loc.saved_darts["d2"]["pos"], 656.658273636061)
        self.assertAlmostEqual(Loc.saved_darts["d3"]["pos"], 247.517908029376)
        self.assertAlmostEqual(Loc.saved_darts["d4"]["pos"], 484.52184735096125)
        self.assertAlmostEqual(Loc.saved_darts["d5"]["pos"], 606.6639115874592)
        self.assertAlmostEqual(Loc.saved_darts["d6"]["pos"], 580.896894915437)

    def test_images_usb_cam_right(self):
        folder_path = (
            r"..\DartDetection\data\test_imgs\250106_usbcam_imgs\imgs_0_to_18\cam_right"
        )

        image_files = sorted([f for f in os.listdir(folder_path)])
        images = [plt.imread(os.path.join(folder_path, f)) for f in image_files]

        Loc = SingleCamLocalize()

        for img in images:
            Loc.new_image(img)

        self.assertAlmostEqual(Loc.saved_darts["d1"]["pos"], 961.1843337382788)
        self.assertAlmostEqual(Loc.saved_darts["d2"]["pos"], 536.1432648703126)
        self.assertAlmostEqual(Loc.saved_darts["d3"]["pos"], 650.3970681707274)
        self.assertAlmostEqual(Loc.saved_darts["d4"]["pos"], 507.66883553411407)
        self.assertAlmostEqual(Loc.saved_darts["d5"]["pos"], 704.5066846428723)
        self.assertAlmostEqual(Loc.saved_darts["d6"]["pos"], 952.0735731554055)


if __name__ == "__main__":
    unittest.main()
