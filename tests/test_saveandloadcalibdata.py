import unittest
from pathlib import Path
import logging
import numpy as np
from dartdetect.calibration import saveandloadcalibdata


class TestSaveAndLoadCalibData(unittest.TestCase):
    def test_load_calibration_data(self):
        # Test case 1: Valid calibration files
        path = Path("data\calibration_matrices")
        calib_dict = saveandloadcalibdata.load_calibration_data(path)

        self.assertIsInstance(calib_dict, dict)
        self.assertIsInstance(calib_dict["l_mtx"], np.ndarray)
        self.assertIsInstance(calib_dict["l_dist"], np.ndarray)
        self.assertIsInstance(calib_dict["r_mtx"], np.ndarray)
        self.assertIsInstance(calib_dict["r_dist"], np.ndarray)
        self.assertIsInstance(calib_dict["R_cl_cr_2d"], np.ndarray)
        self.assertIsInstance(calib_dict["T_cl_cr_2d"], np.ndarray)
        self.assertIsInstance(calib_dict["R_cl_cw_2d"], np.ndarray)
        self.assertIsInstance(calib_dict["T_cl_cw_2d"], np.ndarray)

        # Test case 2: Missing calibration files
        path = Path("data\imgs_cam_calib")

        with self.assertRaises(IOError) as error:
            saveandloadcalibdata.load_calibration_data(path)
            self.assertEqual(
                str(error.exception),
                "Did not find all calibration files in data\imgs_cam_calib, with .npz files: []",
            )
            with self.assertLogs(level=logging.ERROR) as log:
                saveandloadcalibdata.load_calibration_data(path)
            self.assertIn("ERROR:root:Missing calibration file for T_l", log.output)

    def test_load_calibration_matrix(self):
        folder = "data\calibration_matrices"
        cam = "left"
        mtx, dist = saveandloadcalibdata.load_calibration_matrix(folder, cam)

        self.assertIsInstance(mtx, np.ndarray)
        self.assertIsInstance(dist, np.ndarray)


if __name__ == "__main__":
    unittest.main()
