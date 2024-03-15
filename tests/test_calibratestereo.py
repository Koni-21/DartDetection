import unittest
import numpy as np

import dartdetect.calibration.calibrationstereo as dl


class Test_calibratestereo(unittest.TestCase):
    def test_reduce_relations_to_2d(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        T = np.array([1, 2, 3]).reshape(-1, 1)

        R_2d, T_2d = dl.reduce_relations_to_2d(R, T)

        expected_R_2d = np.array([[1, 0], [0, 1]])
        expected_T_2d = np.array([1, 3]).reshape(-1, 1)

        np.testing.assert_array_equal(R_2d, expected_R_2d)
        np.testing.assert_array_equal(T_2d, expected_T_2d)


if __name__ == "__main__":
    unittest.main()
