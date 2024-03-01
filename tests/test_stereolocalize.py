import unittest
import logging
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import dartdetect.stereolocalize as dl


class Teststereolocalizefunctions(unittest.TestCase):
    def test_load_calibration_data(self):
        # Test case 1: Valid calibration files
        path = Path("data\calibration_matrices")
        calib_dict = dl.load_calibration_data(path)

        self.assertIsInstance(calib_dict, dict)
        self.assertIsInstance(calib_dict["l_mtx"], np.ndarray)
        self.assertIsInstance(calib_dict["l_dist"], np.ndarray)
        self.assertIsInstance(calib_dict["r_mtx"], np.ndarray)
        self.assertIsInstance(calib_dict["r_dist"], np.ndarray)
        self.assertIsInstance(calib_dict["R_l"], np.ndarray)
        self.assertIsInstance(calib_dict["T_l"], np.ndarray)
        self.assertIsInstance(calib_dict["R_cl_cw_2d"], np.ndarray)
        self.assertIsInstance(calib_dict["T_cl_cw_2d"], np.ndarray)

        # Test case 2: Missing calibration files
        path = Path("data\imgs_cam_calib")

        with self.assertRaises(IOError) as error:
            dl.load_calibration_data(path)
            self.assertEqual(
                str(error.exception),
                "Did not find all calibration files in data\imgs_cam_calib, with .npz files: []",
            )
            with self.assertLogs(level=logging.ERROR) as log:
                dl.load_calibration_data(path)
            self.assertIn("ERROR:root:Missing calibration file for T_l", log.output)

    def test_projectionmatrics(self):
        l_mtx = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]])
        r_mtx = np.array([[2, 0, 1], [0, 2, 3], [0, 0, 1]])
        R_l2d = np.array([[1, 0], [0, 1]])
        T_l2d = np.array([[1], [3]])

        Pl, Pr = dl.projectionmatrics(l_mtx, r_mtx, R_l2d, T_l2d)

        expected_Pl = np.array([[1, 2, 0], [0, 1, 0]])
        expected_Pr = np.array([[2, 1, 5], [0, 1, 3]])

        np.testing.assert_array_equal(Pl, expected_Pl)
        np.testing.assert_array_equal(Pr, expected_Pr)

    def test_reduce_relations_to_2d(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        T = np.array([1, 2, 3]).reshape(-1, 1)

        R_2d, T_2d = dl.reduce_relations_to_2d(R, T)

        expected_R_2d = np.array([[1, 0], [0, 1]])
        expected_T_2d = np.array([1, 3]).reshape(-1, 1)

        np.testing.assert_array_equal(R_2d, expected_R_2d)
        np.testing.assert_array_equal(T_2d, expected_T_2d)

    def test_combine_rt_homogen(self):
        # Test case 1: Valid inputs
        R = np.array([[1, 0], [0, 1]])
        T = np.array([[1], [2]])
        expected_RT = np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])

        RT = dl.combine_rt_homogen(R, T)

        np.testing.assert_array_equal(RT, expected_RT)

    def test_arrow_img_to_hit_idx_via_lin_fit(self):
        arrow_img = np.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            ]
        )

        distance = 5
        hitpoint, m, b = dl.arrow_img_to_hit_idx_via_lin_fit(
            arrow_img, distance, debug=False
        )

        expected_hitpoint = 4.5
        expected_m = 0
        expected_b = 4.5

        self.assertAlmostEqual(hitpoint, expected_hitpoint)
        self.assertAlmostEqual(m, expected_m)
        self.assertAlmostEqual(b, expected_b)

    def test_DLT(self):
        """
        This test verifies that the DLT method correctly computes the 3D point
        coordinates given the 2D image points and camera matrices.

        The test case sets up the following inputs:
        - Pl: Left camera matrix
        - Pr: Right camera matrix
        - ul: Left image point
        - ur: Right image point

        The expected output is a 2D point coordinate in homogenous coordinates.
        """
        Pl = np.array([[1, 2, 0], [0, 1, 0]])
        Pr = np.array([[2, 1, 5], [0, 1, 3]])
        ul = 1.5
        ur = 2.5

        x = dl.DLT(Pl, Pr, ul, ur)

        expected_x = np.array([[0.5], [-1], [1]])
        np.testing.assert_array_almost_equal(x, expected_x)

    def test_Cl_to_Cw(self):
        """
        This test verifies that the Cl_to_Cw method correctly converts the given Cl vector
        to the corresponding Cw vector using the transformation matrix tr_cl_cw.
        """
        tr_cl_cw = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]])
        cl = np.array([1, 2])

        cr = dl.C1_to_Cw(tr_cl_cw, cl)

        expected_cr = np.array([3, 5])
        np.testing.assert_array_equal(cr, expected_cr)


class TestStereoLocalize(unittest.TestCase):
    def setUp(self):
        # Set up calibration data
        self.calib_dict = {
            "l_mtx": np.array([[1000, 0, 500], [0, 700, 0], [0, 0, 1]]),
            "l_dist": np.array([0, 0, 0, 0, 0]),
            "r_mtx": np.array([[1100, 0, 550], [0, 900, 450], [0, 0, 1]]),
            "r_dist": np.array([0, 0, 0, 0, 0]),
            "R_l": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            "T_l": np.array([[-45], [0], [45]]),
            "R_cl_cw_2d": np.array([[0.7, 0.7], [-0.7, 0.7]]),
            "T_cl_cw_2d": np.array([[32], [32]]),
        }

        # Create DartDetect instance
        self.dart_detect = dl.StereoLocalize(self.calib_dict)

    def test_init(self):
        self.assertEqual(self.dart_detect.calib_dict, self.calib_dict)

        expected_pl = np.array([[1000, 500, 0], [0, 1, 0]])
        expected_pr = np.array([[550, -1100, -24750], [1, 0, 45]])
        expected_tr_c1_cw = np.array([[0.7, 0.7, 32], [-0.7, 0.7, 32], [0, 0, 1]])

        np.testing.assert_array_almost_equal(self.dart_detect.pl, expected_pl)
        np.testing.assert_array_almost_equal(self.dart_detect.pr, expected_pr)
        np.testing.assert_array_almost_equal(
            self.dart_detect.tr_c1_cw, expected_tr_c1_cw
        )

    def test_Cu_to_Cw(self):
        Cul = 500
        Cur = 550

        expected_Cw = np.array([-0.5, -0.5])

        np.testing.assert_array_almost_equal(
            self.dart_detect.Cu_to_Cw(Cul, Cur), expected_Cw
        )

    def test_get_dartpoint_from_Cu(self):
        # Test case 1: dart hit bullseye
        Cul = 500
        Cur = 560
        expected_point = 50

        point = self.dart_detect.get_dartpoint_from_Cu(Cul, Cur)

        self.assertEqual(point, expected_point)

        # Test case 2: Missed the board
        Cul = 1000
        Cur = 1100
        expected_point = 0

        point = self.dart_detect.get_dartpoint_from_Cu(Cul, Cur)

        self.assertEqual(point, expected_point)

    def test_plot_dartposition(self):

        # camera coordinates for bulls eye
        Cul = 500
        Cur = 560
        nr = "Dart Bullseye"
        self.dart_detect.plot_dartboard_emtpy()
        self.dart_detect.plot_dartposition(Cul, Cur, nr)

        # camera coordinates for double 20 (up)
        Cul, Cur = 280, 770
        nr = "Dart d20"
        self.dart_detect.plot_dartposition(Cul, Cur, nr)
        # camera coordinates for double 3 (down)
        self.dart_detect.plot_dartposition(815, 160, "Dart d3")
        # camera coordinates for double 11 (left)
        self.dart_detect.plot_dartposition(150, 350, "Dart d11")
        # camera coordinates for double 6 (right)
        fig = self.dart_detect.plot_dartposition(700, 950, "Dart d6")
        # plt.show()
        self.assertEqual(fig.get_figwidth(), 10.0)


if __name__ == "__main__":
    unittest.main()
