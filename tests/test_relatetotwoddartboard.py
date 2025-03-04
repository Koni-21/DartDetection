import unittest
import numpy as np
import dartdetect.calibration.relatetotwoddartboard as relatetotwoddartboard
import dartdetect.calibration.saveandloadcalibdata as sl_calib


class TestCalculateClCw(unittest.TestCase):
    def test_calculate_cl_cw(self):
        ref_darts_cw = {
            "center": [0, 0],
            "x2y0cm": [2, 0],
            "x10y0cm": [10, 0],
            "x0y10cm": [0, 10],
        }
        ref_darts_cl = {
            "center": [1.25, -45.33],
            "x2y0cm": [-0.14, -46.84],
            "x10y0cm": [-5.66, -52.84],
            "x0y10cm": [8.50, -52.31],
        }

        expected_R_cl_cw = np.array([[0.682592, 0.730799], [-0.730799, 0.682592]])
        expected_T_cl_cw = np.array([32.273897, 31.855407])

        R_cl_cw, T_cl_cw = relatetotwoddartboard.calculate_cl_cw(
            ref_darts_cw, ref_darts_cl
        )

        np.testing.assert_array_almost_equal(R_cl_cw, expected_R_cl_cw)
        np.testing.assert_array_almost_equal(T_cl_cw, expected_T_cl_cw)

    def test_Cl_Cw_angle_from_two_vectors(self):
        cw1 = np.array([0, 0])
        cw2 = np.array([0, 10])
        cl1 = np.array([1, -45])
        cl2 = np.array([0, -46])

        expected_angle = 45

        angle = relatetotwoddartboard.Cl_Cw_angle_from_two_vectors(cw1, cw2, cl1, cl2)

        self.assertAlmostEqual(angle, expected_angle)


class TestCuToCl(unittest.TestCase):
    def setUp(self):
        # Create temp directory using unittest's temporary directory
        import tempfile
        import pathlib

        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)  # Ensure cleanup after test

        tmp_path = pathlib.Path(tmp_dir.name)
        calib_path = tmp_path / "calibration_path"
        calib_path.mkdir(exist_ok=True)

        calib_dict = {
            "l_mtx": np.array([[1000, 0, 500], [0, 700, 0], [0, 0, 1]]),
            "l_dist": np.array([0, 0, 0, 0, 0]),
            "r_mtx": np.array([[1100, 0, 550], [0, 900, 450], [0, 0, 1]]),
            "r_dist": np.array([0, 0, 0, 0, 0]),
            "R_cl_cr_2d": np.array([[0, -1], [1, 0]]),
            "T_cl_cr_2d": np.array([[-45], [45]]),
            "R_cl_cw_2d": np.array([[0.7, 0.7], [-0.7, 0.7]]),
            "T_cl_cw_2d": np.array([[32], [32]]),
        }

        # Create DartDetect instance
        sl_calib.save_calibration_matrix(
            calib_path, calib_dict["l_mtx"], calib_dict["l_dist"], "left"
        )
        sl_calib.save_calibration_matrix(
            calib_path,
            calib_dict["r_mtx"],
            calib_dict["r_dist"],
            "right",
        )
        sl_calib.save_steroecalibration(
            calib_path,
            calib_dict["R_cl_cr_2d"],
            calib_dict["T_cl_cr_2d"],
        )
        sl_calib.save_transformation_cl_cw(
            calib_path,
            calib_dict["R_cl_cw_2d"],
            calib_dict["T_cl_cw_2d"],
        )
        self.calib_path = calib_path

    def test_call(self):
        ul = 640
        ur = 650
        expected_v_cl = np.array([-6.786355, -48.473968])

        cu_to_cl = relatetotwoddartboard.CuToCl(self.calib_path)
        v_cl = cu_to_cl(ul, ur)

        np.testing.assert_array_almost_equal(v_cl, expected_v_cl)

    def test_dict_of_cu_to_cl(self):
        dict_of_cu = {
            "key1": [640, 660],
            "key2": [520, 740],
        }

        expected_dict_of_cl = {
            "key1": [-6.83432, -48.816568],
            "key2": [-1.051821, -52.591049],
        }

        cu_to_cl = relatetotwoddartboard.CuToCl(self.calib_path)
        dict_of_cl = cu_to_cl.dict_of_cu_to_cl(dict_of_cu)

        for key in dict_of_cl:
            np.testing.assert_array_almost_equal(
                dict_of_cl[key], expected_dict_of_cl[key]
            )


if __name__ == "__main__":
    unittest.main()
