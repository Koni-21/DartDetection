import unittest
import logging
import numpy as np
from pathlib import Path

import dartdetect.dartdetection as dd


class TestLoadCalibrationData(unittest.TestCase):
    def test_load_calibration_data(self):
        # Test case 1: Valid calibration files
        path = Path("data\calibration_matrices")
        calib_dict = dd.load_calibration_data(path)

        self.assertIsInstance(calib_dict, dict)
        self.assertIsInstance(calib_dict["l_mtx"], np.ndarray)
        self.assertIsInstance(calib_dict["l_dist"], np.ndarray)
        self.assertIsInstance(calib_dict["r_mtx"], np.ndarray)
        self.assertIsInstance(calib_dict["r_dist"], np.ndarray)
        self.assertIsInstance(calib_dict["R_l"], np.ndarray)
        self.assertIsInstance(calib_dict["T_l"], np.ndarray)

        # Test case 2: Missing calibration files
        path = Path("data\imgs_cam_calib")
        calib_dict = dd.load_calibration_data(path)

        self.assertIsInstance(calib_dict, dict)
        self.assertIsNone(calib_dict["l_mtx"])
        self.assertIsNone(calib_dict["l_dist"])
        self.assertIsNone(calib_dict["r_mtx"])
        self.assertIsNone(calib_dict["r_dist"])
        self.assertIsNone(calib_dict["R_l"])
        self.assertIsNone(calib_dict["T_l"])

        with self.assertLogs(level=logging.ERROR) as log:
            dd.load_calibration_data(path)

        self.assertIn("ERROR:root:Missing calibration file for T_l", log.output)


if __name__ == "__main__":
    unittest.main()
