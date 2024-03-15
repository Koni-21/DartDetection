import pathlib
import logging
import numpy as np

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
        R_cl_cw_2d: np.array, 2x2, rotation matrix from the left camera
            coordinate system Cl to the 2D world coordinate system
        T_cl_cw_2d: np.array, 2x1, translation vector from the left camera
            coordinate system Cl to the 2D world coordinate system
    """

    path = pathlib.Path(path)
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
        "R_cl_cw_2d": None,
        "T_cl_cw_2d": None,
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
            if ("stereo" and "cr") in str(calib_file):
                calib_dict["R_l"], calib_dict["T_l"] = [data[i] for i in ("R", "T")]
            if ("stereo" and "cw") in str(calib_file):
                calib_dict["R_cl_cw_2d"], calib_dict["T_cl_cw_2d"] = [
                    data[i] for i in ("R", "T")
                ]

    [
        LOGGER.error(f"Missing calibration file for {key}")
        for key, value in calib_dict.items()
        if value is None
    ]

    if None in [x for x in calib_dict.values() if x is None]:
        raise IOError(
            f"Did not find all calibration files in {path}, with .npz files: {calib_files}."
        )

    return calib_dict


def load_calibration_matrix(folder, cam):
    filename = pathlib.Path.joinpath(pathlib.Path(folder), f"calib_cam_{cam}.npz")
    with np.load(filename) as X:
        mtx, dist = [X[i] for i in ("mtx", "dist")]
    return mtx, dist


def save_calibration_matrix(folder, mtx, dist, cam):
    filename = pathlib.Path.joinpath(pathlib.Path(folder), f"calib_cam_{cam}.npz")
    np.savez(filename, mtx=mtx, dist=dist)


def save_steroecalibration(folder, R, T):
    filename = pathlib.Path.joinpath(pathlib.Path(folder), f"stereo_RT_cl_cr.npz")
    np.savez(filename, R=R, T=T)


def save_transformation_cl_cw(folder, R_cl_cw, T_cl_cw):
    filename = pathlib.Path.joinpath(pathlib.Path(folder), "stereo_RT_cl_cw.npz")
    np.savez(filename, R=R_cl_cw, T=T_cl_cw)
