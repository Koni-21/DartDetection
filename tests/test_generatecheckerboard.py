import unittest
import numpy as np
import cv2

import dartdetect.calibration.generatecheckerboard as gencheck


class TestFindSquareSize(unittest.TestCase):
    def test_find_square_size(self):
        pixels_h = 640
        pixels_v = 480
        diag = 10.0

        expected_possible_dimensions = {
            0: (1.0, 80, 6.0, 8.0),
            1: (2.0, 160, 3.0, 4.0),
        }
        expected_cm_dimensions = (6.0, 8.0)

        possible_dimensions, cm_dimensions = gencheck.find_square_size(
            pixels_h, pixels_v, diag
        )

        self.assertEqual(possible_dimensions, expected_possible_dimensions)
        self.assertEqual(cm_dimensions, expected_cm_dimensions)

    def test_generate_chessboard(self):
        pixels_h = 4
        rows = 2
        columns = 2

        expected_chessboard = np.array(
            [
                [
                    [255, 255, 255],
                    [255, 255, 255],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [255, 255, 255],
                    [255, 255, 255],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [255, 255, 255],
                    [255, 255, 255],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [255, 255, 255],
                    [255, 255, 255],
                ],
            ],
            dtype=np.uint8,
        )

        generated_chessboard = gencheck.generate_chessboard(pixels_h, rows, columns)

        np.testing.assert_array_almost_equal(generated_chessboard, expected_chessboard)


if __name__ == "__main__":
    unittest.main()
