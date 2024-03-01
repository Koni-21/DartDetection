import unittest
from dartdetect.dartboardgeometry import DartboardGeometry


class TestDartboardGeometry(unittest.TestCase):
    def setUp(self):
        self.dartboard = DartboardGeometry()

    def test_get_dartpoint_from_cart_coordinates(self):
        # Test case 1: Dart hits the bullseye
        x = 0
        y = 0
        expected_point = 50

        point = self.dartboard.get_dartpoint_from_cart_coordinates(x, y)

        self.assertEqual(point, expected_point)

        # Test case 2: Dart hits the triple 6 field
        x = 10
        y = 0
        expected_point = 18

        point = self.dartboard.get_dartpoint_from_cart_coordinates(x, y)

        self.assertEqual(point, expected_point)

        # Test case 3: Dart hits the 20 field
        x = 0
        y = 15
        expected_point = 20

        point = self.dartboard.get_dartpoint_from_cart_coordinates(x, y)

        self.assertEqual(point, expected_point)

        # Test case 4: Dart hits outside the dartboard
        x = 30
        y = 30
        expected_point = 0

        point = self.dartboard.get_dartpoint_from_cart_coordinates(x, y)

        self.assertEqual(point, expected_point)

    def test_plot_dartboard_emtpy(self):
        # Test case: Check if the plot is generated without errors
        fig = self.dartboard.plot_dartboard_emtpy()

        self.assertIsNotNone(fig)

    def test_plot_dartposition(self):
        # Test case: Check if the dart position is plotted correctly
        x = 5
        y = 5
        nr = "Dart 1"
        color = "red"

        fig = self.dartboard.plot_dartposition(x, y, nr, color)

        self.assertIsNotNone(fig)

    def test_clear_plot(self):
        # Test case: Check if the plot is cleared without errors
        self.dartboard.clear_plot()

        # No assertion, just checking for any exceptions


if __name__ == "__main__":
    unittest.main()
