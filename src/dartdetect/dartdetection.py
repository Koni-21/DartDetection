import pathlib
import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import svd
from .dartboard_geometry import dartboard_geometry

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def load_calibration_data(path=pathlib.Path.cwd()) -> dict:
    """Loads the calibration data from a file.
    Args:
        path: str, path to the file
    Returns:
        calib_dict: dict, dictionary containing the calibration data
            l_mtx: np.array, 3x3, intrinsic matrix of the left camera
            l_dist: np.array, 1x5, distortion coefficients of the left camera
            r_mtx: np.array, 3x3, intrinsic matrix of the right camera
            r_dist: np.array, 1x5, distortion coefficients of the right camera
            R_l: np.array, 3x3, rotation matrix from the left camera to the right camera
            T_l: np.array, 3x1, translation vector from the left camera to the right camera
    """

    file_list = pathlib.Path.iterdir(path)
    # Find the part of the filename in files
    calib_files = [file for file in file_list if str(file).endswith(".npz")]

    calib_dict = {
        "l_mtx": None,
        "l_dist": None,
        "r_mtx": None,
        "r_dist": None,
        "R_l": None,
        "T_l": None,
    }
    for calib_file in calib_files:
        with np.load(calib_file) as data:
            if "left" in str(calib_file):
                calib_dict["l_mtx"], calib_dict["l_dist"] = [
                    data[i] for i in ("mtx", "dist")
                ]
            if "right" in str(calib_file):
                calib_dict["r_mtx"], calib_dict["r_dist"] = [
                    data[i] for i in ("mtx", "dist")
                ]
            if "stereo" in str(calib_file):
                calib_dict["R_l"], calib_dict["T_l"] = [data[i] for i in ("R", "T")]

    [
        LOGGER.error(f"Missing calibration file for {key}")
        for key, value in calib_dict.items()
        if value is None
    ]

    return calib_dict


def projectionmatrics(l_mtx, r_mtx, R_l2d, T_l2d):
    """Calculates the projection matrices of the left and right camera.
    Args:
        l_mtx: np.array, 3x3, intrinsic matrix of the left camera
        r_mtx: np.array, 3x3, intrinsic matrix of the right camera
        R_l2d: np.array, 3x3, rotation matrix from the left camera to the 2D world coordinate system
        T_l2d: np.array, 3x1, translation vector from the left camera to the 2D world coordinate system
    Returns:
        Pl: np.array, 3x4, projection matrix of the left camera
        Pr: np.array, 3x4, projection matrix of the right camera
    """
    l_mtx2d = np.array([[l_mtx[0, 0], l_mtx[0, 2], 0], [0, 1, 0]])
    tr_l = np.eye(3)
    Pl = l_mtx2d @ tr_l
    r_mtx2d = np.array([[r_mtx[0, 0], r_mtx[0, 2], 0], [0, 1, 0]])
    tr_r = np.vstack((np.concatenate([R_l2d, T_l2d], axis=-1), [0, 0, 1]))
    Pr = r_mtx2d @ tr_r
    return Pl, Pr


def arrow_img_to_hit_idx_via_lin_fit(arrow_img, distance, debug=False):
    """Calculates the hitpoint coordinate of an arrow in an image via linear regression.

    Args:
        arrow_img: np.array, 2D, binary image of the arrow mask
        distance: int, distance in pixels from the lower image edge to
            the upper dartboard edge
        debug: bool, if True, the regression line is plotted

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

    if debug:
        plt.scatter(positions_xy[1], positions_xy[0])
        plt.plot(positions_xy[1], m * positions_xy[0] + b, color="red")

    hitpoint = m * (positions_xy[1].max() + distance) + b

    return hitpoint, m, b


def DLT(Pl, Pr, ul, ur):
    """Direct Linear Transformation (DLT) algorithm to solve for the 2D point X
    Args:
        Pl: np.array, 3x4, projection matrix of the left camera
        Pr: np.array, 3x4, projection matrix of the right camera
        ul: float, x-coordinate of the point in the left image
        ur: float, x-coordinate of the point in the right image
    Returns:
        x: np.array, 3x1, 2D point in the world coordinate system in homogeneous coordinates
    """
    A = ([Pl[0, :] - ul * Pl[1, :], Pr[0, :] - ur * Pr[1, :]],)
    A = np.array(A).reshape(2, 3)
    B = A.transpose() @ A
    _, _, Vh = svd(B, full_matrices=False)
    x = (Vh[-1, :] / Vh[-1, -1]).reshape(3, 1)
    return x


def Cl_to_Cw(tr_cl_cw, cl):
    """Transforms a point from Cl to Cw.
    Args:
        tr_cl_cw: np.array, 3x3, transformation matrix from Cl to Cw
        cl: np.array, 2x1, point in Cl
    Returns:
        cr: np.array, 2x1, point in Cw
    """
    x = tr_cl_cw
    cr = x @ np.vstack((cl.reshape(2, 1), [1]))
    return cr[:2].flatten()


class DartDetect(dartboard_geometry):
    def __init__(self, matrizen, debug=False):
        pass
        # detect image funktion
        # ...

    # def Cu_to_Cw(Cul, Cur):
    #     """ Transforms a point from Cu to Cw.
    #     Args:
    #         Cul: float, x-coordinate of the point in the left image
    #         Cur: float, x-coordinate of the point in the right image
    #         tr_cl_cw: np.array, 3x3, transformation matrix from Cl to Cw
    #         pl: np.array, 3x4, projection matrix of the left camera
    #         pr: np.array, 3x4, projection matrix of the right camera
    #     Returns:
    #         Cw: np.array, 2x1, point in Cw
    #     """
    #     Cl = DLT(pl, pr, Cul, Cur)[:2].flatten()
    #     return -1* Cl_to_Cw(tr_cl_cw, Cl)
