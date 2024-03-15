import numpy as np
import matplotlib.pyplot as plt

import cv2
import ultralytics

import dartdetect.calibration.saveandloadcalibdata as sl_calib


def arrow_img_to_hit_idx_via_lin_fit(arrow_img, distance):
    """Calculates the hitpoint coordinate of an arrow in an image via linear regression.

    Args:
        arrow_img: np.array, 2D, binary image of the arrow mask
        distance: int, distance in pixels from the lower image edge to
            the upper dartboard edge

    returns:
        hitpoint: float, x-coordinate of the hitpoint
        m: float, slope of the regression line
        b: float, x-intercept of the regression line

    note:
        example arrow_img, size 10x5:
        0 0 0 1 1 1 1 0 0 0
        0 0 0 1 1 1 1 0 0 0
        0 0 0 0 1 1 0 0 0 0
        0 0 0 0 1 1 0 0 0 0
        0 0 0 0 1 1 0 0 0 0
    """
    positions_xy = arrow_img.T.nonzero()

    # fit straight line via regression:
    m, b = np.polyfit(positions_xy[1], positions_xy[0], 1)
    hitpoint = m * (positions_xy[1].max() + distance) + b

    return hitpoint, m, b


class DartLocalize:
    """
    Class for localizing the hitpoints of darts in an image.
    """

    def __init__(self, img, camera, calibration_path, model_path, distance):
        """
        Args:
            img: np.array, 2D, example image of one camera view
            camera: str, camera type, e.g. "left" or "right"
            calibration_path: str, path to the calibration data
            model_path: str, path to the YOLO model
            distance: int, distance in pixels from the lower image edge to
                the upper dartboard edge
        """

        self.model = ultralytics.YOLO(model_path)
        self.distance = distance
        self.mtx, self.dist = sl_calib.load_calibration_matrix(calibration_path, camera)
        # calculate new camera matrix
        h, w = img.shape[:2]
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

    def predict(self, img):
        """
        Predicts the location of darts in the given image.

        Args:
            img: The input image for dart localization.

        Returns:
            The predicted location of darts in the image.
        """
        return self.model.predict(source=img, conf=0.50, verbose=False)

    def __call__(self, img):
        """
        Localizes the hitpoints of all arrows in an image.

        Args:
            img: np.array, 2D, binary image of the arrow mask

        Returns:
            hitpoint: list of float, x-coordinate of the hitpoint
            line_params: list of tuple, (m, b), slope and x-intercept of the regression line
            dart_mask: list of np.array, 2D, binary image of the dart mask

        """

        dst_img = self.undistort(img)
        results = self.predict(dst_img)

        coords_cu1 = []
        line_params = []
        for result in results[0]:
            dart_mask = np.array(result.masks.data[0])
            point_coord, m, b = arrow_img_to_hit_idx_via_lin_fit(
                dart_mask, self.distance
            )
            # the network predicts the image in different dimensions
            reshape_correction = np.shape(img)[1] / np.shape(dart_mask)[1]
            dart_coord = point_coord * reshape_correction
            b = b * reshape_correction
            m = m * reshape_correction
            line_params.append((m, b))
            coords_cu1.append(dart_coord)

        return coords_cu1, line_params, dart_mask


if __name__ == "__main__":
    import pathlib
    import scipy.ndimage

    folder_test_images = "data/imgs_dartboard_calib"
    path = pathlib.Path(folder_test_images)
    predict_imgs = [img for img in path.iterdir() if img.is_file()]
    predict_imgs.sort()

    img = cv2.imread(str(predict_imgs[0]))
    Darts_Left = DartLocalize(
        img,
        "left",
        pathlib.Path("data/calibration_matrices"),
        pathlib.Path("data/segmentation_model/yolov8_seg_dart.pt"),
        0,
    )

    # Segmentation mask is inaccurate! --> ai needs improvement or replacement
    pos, line_params, dart_mask = Darts_Left(img)
    m, b = line_params[0]
    print(f"{pos=}, {m=}, {b=}")
    dart_mask = scipy.ndimage.zoom(dart_mask, 1240 / 1248)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(dart_mask, alpha=0.5, cmap="Reds")
    y = np.arange(0, np.shape(img)[0])

    plt.plot(m * y + b, y, "r", linewidth=2, alpha=0.5)
    plt.show()
