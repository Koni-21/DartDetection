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

    if ref_darts_cw.keys() != ref_darts_cl.keys():
        raise (KeyError("The reference points are the same"))
    if ref_darts_cw.get("center", None) is None:
        raise (KeyError("The key 'center' must be defined in both dictionaries"))

    angles = []
    # Generate all binomial pairs of keys
    pairs = list(combinations(ref_darts_cw.keys(), 2))
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
    LOGGER.info(f"{tr_cl_cw=}, \n{T_cl_cw=},\n {R_cl_cw=}")
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


if __name__ == "__main__":
    import dartdetect.calibration.saveandloadcalibdata as sl_calib

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
    calculate_cl_cw(ref_darts_cw, ref_darts_cl)

    calib_dict = sl_calib.load_calibration_data(path="data/calibration_matrices")
    Cu_to_Cl = CuToCl(calib_dict)
    print(Cu_to_Cl(640, 640))

    ref_darts_cu = {
        "center": [640, 640],
        "x2y0cm": [660, 640],
        "x10y0cm": [740, 640],
        "x0y10cm": [640, 740],
    }
    print(Cu_to_Cl.dict_of_cu_to_cl(ref_darts_cu))