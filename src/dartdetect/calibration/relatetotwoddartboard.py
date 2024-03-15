"""
Module for calculating the transformation matrix from the 
camera lefts (Cl) coordinate system to the dartboards 
coordinate system (Cw). --> stereo_RT_cl_cw.npz

This is done by using known darts as reference points on the dartboard.
"""

import logging
from itertools import combinations
import numpy as np

from dartdetect import stereolocalize


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def Cl_Cw_angle_from_two_vectors(cw1, cw2, cl1, cl2):
    """Calculates the angle between two vectors in Cw and Cl.

    Args:
        cw1: list or np.array, 2x1, vector in Cw
        cw2: list or np.array, 2x1, vector in Cw
        cl1: list or np.array, 2x1, vector in Cl
        cl2: list or np.array, 2x1, vector in Cl

    Returns:
        angle: float, angle between the vectors in degree
    """
    cw1 = np.array(cw1).reshape(2, 1)
    cw2 = np.array(cw2).reshape(2, 1)
    delta_cw = cw2 - cw1
    cl1 = np.array(cl1).reshape(2, 1)
    cl2 = np.array(cl2).reshape(2, 1)
    delta_cl = cl2 - cl1
    # https://en.wikipedia.org/wiki/Dot_product#/media/File:Inner-product-angle.svg
    angle_cl_cw = np.arccos(
        np.dot(delta_cw.T, delta_cl)
        / (np.linalg.norm(delta_cw) * np.linalg.norm(delta_cl))
    )

    return 180 - np.rad2deg(angle_cl_cw)[-1][-1]


def calculate_cl_cw(ref_darts_cw, ref_darts_cl):
    """Calculates the rotation and translation matrix from the world coordinate
    system Cw to the left camera coordinate system Cl.
    Both input dictionaries must have the same keys and must contain the
    center coordinate of the Dartboard cw = [0,0].

    Args:
        ref_darts_cw: dict, dictionary containing the reference points in Cw
        ref_darts_cl: dict, dictionary containing the reference points in Cl


    Returns:
        R_cl_cw: np.array, 2x2, rotation matrix from Cw to Cl
        T_cl_cw: np.array, 2x1, translation vector from Cw to Cl

    Note:
        the approach is not optimal and a bit complicated. Better approach is to
        solve the dot product of cw = RT $\cdot$ cl with a least squares approach.
    """
    keys_cw, keys_cl = list(ref_darts_cw.keys()), list(ref_darts_cl.keys())

    keys_cw.sort(), keys_cl.sort()
    if keys_cw != keys_cl:
        raise (
            KeyError(
                f"The reference points are not the same: "
                f"Missing Keys in cl{[missing_key for missing_key in keys_cw if missing_key not in keys_cl]}, "
                f"Missing Keys in cw{[missing_key for missing_key in keys_cl if missing_key not in keys_cw]}, "
            )
        )
    if ref_darts_cw.get("center", None) is None:
        raise (KeyError("The key 'center' must be defined in both dictionaries"))

    angles = []
    # Generate all binomial pairs of keys
    pairs = list(combinations(keys_cw, 2))
    for pos1, pos2 in pairs:
        angle = Cl_Cw_angle_from_two_vectors(
            ref_darts_cw[pos1],
            ref_darts_cw[pos2],
            ref_darts_cl[pos1],
            ref_darts_cl[pos2],
        )
        LOGGER.info(f"Angle between {pos1} and {pos2} in Cw: {angle:.3f} °")
        angles.append(angle)

    LOGGER.info(f"{np.mean(angles)=}, {np.std(angles)=}")
    dist = np.linalg.norm(ref_darts_cl["center"])
    LOGGER.info(f"distance: {dist:.4f} cm")
    # phi1 is the error angle between main ray of the camera and the
    # center of the board --> with a good adjustment of the camera this
    # angle should be < 1°.
    phi_1 = np.rad2deg(np.arctan(ref_darts_cl["center"][0] / ref_darts_cl["center"][1]))
    phi_2 = np.mean(angles) + phi_1
    LOGGER.info(f"{phi_1=}, {phi_2=}")
    # calculate the translation vector from Cw to Cl
    x, y = -dist * np.sin(np.deg2rad(phi_2)), -dist * np.cos(np.deg2rad(phi_2))
    pos_cl = [x, y]
    LOGGER.info(f"translation: {pos_cl}")

    phi_2 = -1 * np.mean(angles)
    tr_cl_cw = np.array(
        [
            [np.cos(np.deg2rad(phi_2)), -np.sin(np.deg2rad(phi_2)), -1 * pos_cl[0]],
            [np.sin(np.deg2rad(phi_2)), np.cos(np.deg2rad(phi_2)), -1 * pos_cl[1]],
            [0, 0, 1],
        ]
    )

    T_cl_cw = tr_cl_cw[:2, 2]
    R_cl_cw = tr_cl_cw[0:2, 0:2]
    LOGGER.info(f"{tr_cl_cw=}, \n {T_cl_cw=},\n {R_cl_cw=}")
    return R_cl_cw, T_cl_cw


class CuToCl(stereolocalize.StereoLocalize):
    """
    Class for converting coordinates from  the camera pixel coordinate
    systems (Cul, Cur) to the camera lefts (Cl) coordinate system.
    Inherits from the StereoLocalize class.
    """

    def __init__(self, calib_dict):
        super().__init__(calib_dict)

    def __call__(self, ul, ur):
        """
        Converts the given camera pixel coordinates (ul, ur) to
        camera left (cl) coordinates.

        Args:
            ul (float): Camera pixel coordinate of the left camera.
            ur (float): Camera pixel coordinate of the right camera.

        Returns:
            numpy.ndarray (2x1): Coordinates in camera lefts coordinate system.
        """
        v_cl = stereolocalize.DLT(self.pl, self.pr, ul, ur)[:2].flatten()
        return v_cl

    def dict_of_cu_to_cl(self, dict_of_cu):
        """
        Converts a dictionary of camera pixel coordinates to camera left coordinates.

        Args:
            dict_of_cu (dict): Dictionary containing camera pixel coordinates as values.

        Returns:
            dict: Dictionary containing camera left coordinates as values.
        """
        dict_of_cl = {}
        for key, value in dict_of_cu.items():
            dict_of_cl[key] = self(value[0], value[1])
        return dict_of_cl


def generate_dict_of_cu_from_images(folder_test_images, DartRight, DartLeft):
    """
    Generate a dictionary of Cu values from a folder of test images.

    Args:
        folder_test_images (str): The path to the folder containing the test images.
        DartRight (dartdetect.dartlocalize.DartLocalize class): A class object
            that takes an image and returns the Cu value for the right dart.
        DartLeft (dartdetect.dartlocalize.DartLocalize class): A class object
            that takes an image and returns the Cu value for the left dart.

    Returns:
        dict: A dictionary where the keys are image names and the values
        are lists of Cu values for the left and right darts.

    Raises:
        ValueError: If the keys of the left and right images are not the same.

    """
    path = pathlib.Path(folder_test_images)
    predict_imgs_l = [
        img for img in path.iterdir() if img.is_file() and "left" in str(img)
    ]
    predict_imgs_r = [
        img for img in path.iterdir() if img.is_file() and "right" in str(img)
    ]

    keys_l = [str(img_name).split("cam_left_")[1][:-4] for img_name in predict_imgs_l]
    keys_r = [str(img_name).split("cam_right_")[1][:-4] for img_name in predict_imgs_r]

    keys_l.sort()
    keys_r.sort()

    if keys_l != keys_r:
        raise (
            ValueError(
                f"The keys of the images are not the same: \n {keys_l=}, \n {keys_r=}"
            )
        )

    Cu_dict = {}
    for key in keys_l:
        filename_l = [name for name in predict_imgs_l if key in str(name)][0]
        filename_r = [name for name in predict_imgs_r if key in str(name)][0]
        ul, _, _ = DartLeft(cv2.imread(filename_l))
        ur, _, _ = DartRight(cv2.imread(filename_r))
        Cu_dict[key] = [ul[0], ur[0]]

    return Cu_dict


if __name__ == "__main__":
    import pathlib
    import cv2

    import dartdetect.calibration.saveandloadcalibdata as sl_calib
    import dartdetect.dartlocalize as dl

    ref_darts_cw = {
        "center": [0, 0],
        "x2y0cm": [2, 0],
        "x0y2cm": [0, 2],
        "x10y0cm": [10, 0],
        "x0y10cm": [0, 10],
        "s20id": [-2.26067, 15.68795],
        "s20it": [-1.37209, 9.52165],
        "s11id": [-15.68795, 2.26067],
        "s11it": [-9.52165, 1.37209],
        "s3id": [-2.26067, -15.68795],
        "s3it": [-1.37209, -9.52165],
        "s6id": [15.68795, 2.26067],
        "s6it": [9.52165, 1.37209],
        "t10": [9.41, -4.69],
    }

    folder_test_images = "data/imgs_dartboard_calib"
    DartLeft = dl.DartLocalize(
        "left",
        pathlib.Path("data/calibration_matrices"),
        pathlib.Path("data/segmentation_model/yolov8_seg_dart.pt"),
    )
    DartRight = dl.DartLocalize(
        "right",
        pathlib.Path("data/calibration_matrices"),
        pathlib.Path("data/segmentation_model/yolov8_seg_dart.pt"),
    )

    ref_darts_cu = generate_dict_of_cu_from_images(
        folder_test_images, DartRight, DartLeft
    )
    print(ref_darts_cu)
    calib_dict = sl_calib.load_calibration_data(path="data/calibration_matrices")
    Cu_to_Cl = CuToCl(calib_dict)
    ref_darts_cl = Cu_to_Cl.dict_of_cu_to_cl(ref_darts_cu)

    print(calculate_cl_cw(ref_darts_cw, ref_darts_cl))
