import unittest
import pathlib

import numpy as np
import cv2
import dartdetect.dartlocalize as dl


def get_mean_nonzero(img):
    return np.mean(
        np.nonzero(
            np.array(
                cv2.threshold(img, 0.8, 1, cv2.THRESH_BINARY)[1],
                dtype=int,
            )
            - 1.0
        )[1]
    )


def read_img_cv(path):
    img_cv2 = cv2.imread(path)
    img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    img_cv2_float16 = img_cv2_gray.astype(np.float32) / (255)
    return img_cv2_float16


class TestDartLocalize(unittest.TestCase):
    def setUp(self):

        self.camera = "left"
        self.calibration_path = pathlib.Path("data/calibration_matrices")
        self.DartLocal = dl.DartLocalize(self.camera, self.calibration_path)

        self.img0 = read_img_cv(
            "data/test_imgs/250106_usbcam_imgs/imgs_0_to_18/cam_left/img_04_camera_left_t20250106174634636.png"
        )

        self.img1 = read_img_cv(
            "data/imgs_dartboard_calib/t20240217212732931_cam_left_t10.png"
        )

        self.img2 = read_img_cv(
            "data/imgs_dartboard_calib/t20240224144711290_cam_left_x2y0cm.png"
        )

    def test_undistort(self):
        img = self.img1

        img_undist = self.DartLocal.undistort(img)

        result = get_mean_nonzero(img_undist)
        expected_result = 866.5981308411215
        expected_result_distort = 865.5922047702153

        self.assertNotAlmostEqual(result, get_mean_nonzero(img))
        self.assertAlmostEqual(result, expected_result)
        self.assertAlmostEqual(get_mean_nonzero(img), expected_result_distort)

    def test_call(self):
        result1 = self.DartLocal(self.img0)

        assert result1 == None
        result1 = self.DartLocal(self.img1)
        expected_result = 868.7022230203621
        self.assertAlmostEqual(result1, expected_result)


if __name__ == "__main__":
    unittest.main()
