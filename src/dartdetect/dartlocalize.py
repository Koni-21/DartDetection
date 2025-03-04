import numpy as np
import matplotlib.pyplot as plt

import cv2


import dartdetect.calibration.saveandloadcalibdata as sl_calib
from dartdetect.singlecamdartlocalize.singlecamlocalize import SingleCamLocalize


class DartLocalize(SingleCamLocalize):
    """
    Class for localizing the hitpoints of darts in an image.
    """

    def __init__(
        self, camera, calibration_path, w=1240, h=90, distance_to_dartboard_px=0
    ):
        """
        Args:
            img: np.array, 2D, example image of one camera view
            camera: str, camera type, e.g. "left" or "right"
            calibration_path: str, path to the calibration data
            distance: int, distance in pixels from the lower image edge to
                the upper dartboard edge
        """
        super().__init__()
        # self.Localizer = SingleCamLocalize()
        self.distance = distance_to_dartboard_px
        self.mtx, self.dist = sl_calib.load_calibration_matrix(calibration_path, camera)
        # calculate new camera matrix
        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 0, (w, h)
        )

    def undistort(self, img):
        """
        Applies camera distortion correction to the input image.

        Args:
            img: The input image to be undistorted.

        Returns:
            The undistorted image.
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)

    def __call__(self, img):
        """
        Localizes the hitpoints of all arrows in an image.

        Args:
            img: np.array, 2D, binary image of the arrow mask

        Returns:
            hitpoint: float, x-coordinate of the hitpoint

        """

        dst_img = self.undistort(img)
        result = super().new_image(dst_img)
        # self.Localizer.visualize_stream()
        if result:
            pos, angle = result["pos"], result["angle"]
            new_pos_with_distance = pos - np.sin(np.deg2rad(angle)) * self.distance
            return new_pos_with_distance
        else:  # no dart detected
            return None


if __name__ == "__main__":
    import pathlib
    import scipy.ndimage

    Darts_Left = DartLocalize(
        "left",
        pathlib.Path("data/calibration_matrices"),
        distance_to_dartboard_px=10,
    )

    def read_img_cv(path):
        img_cv2 = cv2.imread(path)
        img_cv2_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        img_cv2_float16 = img_cv2_gray.astype(np.float32) / (255)
        return img_cv2_float16

    img0 = read_img_cv(
        "data/test_imgs/250106_usbcam_imgs/imgs_0_to_18/cam_left/img_04_camera_left_t20250106174634636.png"
    )
    img1 = read_img_cv(
        "data/imgs_dartboard_calib/t20240224144711290_cam_left_x2y0cm.png"
    )
    pos = Darts_Left(img0)
    pos = Darts_Left(img1)
    if pos:
        print(f"{pos=}")
    plt.show()
