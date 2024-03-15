import unittest
import pathlib

import numpy as np
import cv2
import dartdetect.dartlocalize as dl


def get_mean_nonzero(img):
    return np.mean(
        np.nonzero(
            np.array(
                cv2.threshold(img, 200, 100, cv2.THRESH_BINARY)[1][:, :, 0],
                dtype=int,
            )
            - 100
        )[1]
    )


class Testdartlocalizefunctions(unittest.TestCase):
    def test_arrow_img_to_hit_idx_via_lin_fit(self):
        arrow_img = np.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            ]
        )

        distance = 5
        hitpoint, m, b = dl.arrow_img_to_hit_idx_via_lin_fit(arrow_img, distance)

        expected_hitpoint = 5.3502024291
        expected_m = 0.10323886639676094
        expected_b = 4.421052631578948

        self.assertAlmostEqual(hitpoint, expected_hitpoint)
        self.assertAlmostEqual(m, expected_m)
        self.assertAlmostEqual(b, expected_b)

        arrow_img = np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        distance = 0

        hitpoint, m, b = dl.arrow_img_to_hit_idx_via_lin_fit(arrow_img, distance)

        expected_hitpoint = 6
        expected_m = 1
        expected_b = 2

        np.testing.assert_array_almost_equal(
            [hitpoint, m, b], [expected_hitpoint, expected_m, expected_b]
        )


class TestDartLocalize(unittest.TestCase):
    def setUp(self):
        filepath = pathlib.Path(
            "data/imgs_dartboard_calib/t20240217212732931_cam_left_t10.png"
        )
        self.img = cv2.imread(str(filepath))
        self.camera = "left"
        self.calibration_path = pathlib.Path("data/calibration_matrices")
        self.model_path = pathlib.Path("data/segmentation_model/yolov8_seg_dart.pt")
        self.DartLocal = dl.DartLocalize(
            self.camera, self.calibration_path, self.model_path
        )

    def test_undistort(self):
        img = self.img

        img_undist = self.DartLocal.undistort(img)

        result = get_mean_nonzero(img_undist)
        expected_result = 866.6039835969538
        expected_result_distort = 865.593914569924

        self.assertNotAlmostEqual(result, get_mean_nonzero(img))
        self.assertAlmostEqual(result, expected_result)
        self.assertAlmostEqual(get_mean_nonzero(img), expected_result_distort)

    def test_predict(self):
        result = self.DartLocal.predict(self.img)[0][0]
        mask = np.array(result.masks.data[0])

        result = np.mean(np.nonzero(mask)[1])
        expected_result = 871.731308411215

        self.assertEqual(np.shape(self.img), (90, 1240, 3))
        self.assertEqual(np.shape(mask), (96, 1248))
        self.assertAlmostEqual(result, expected_result)

        result_scaled = result * np.shape(self.img)[1] / np.shape(mask)[1]
        expected_result_scaled = 866.1432872034508
        self.assertAlmostEqual(result_scaled, expected_result_scaled)

    def test_call(self):
        results, res_line, _ = self.DartLocal(self.img)
        expected_results = [868.5230634086527]
        expected_res_line = (0.01005565992901522, 867.6079983551123)

        np.testing.assert_array_almost_equal(results, expected_results)
        np.testing.assert_array_almost_equal(res_line[-1], expected_res_line)


if __name__ == "__main__":
    unittest.main()
