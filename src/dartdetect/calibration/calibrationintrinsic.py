from pathlib import Path

import numpy as np
import cv2


def calibrate_camera(chess_grid, world_scaling, cam, folder_name, show_output=True):
    """
    Calibrate the camera using the chessboard images in the given folder

    Args:
        chess_grid: tuple of the number of corners in the chessboard
        world_scaling: scaling of the chessboard in cm
        cam: camera name
        folder_name: folder containing the images
        show_output: if True, show the images with the found corners

    Returns:
        mtx: camera matrix
        dist: distortion coefficients

    Note:
        the checkerboard needs a black border

    References:
        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
        https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_grid[0] * chess_grid[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chess_grid[0], 0 : chess_grid[1]].T.reshape(-1, 2)

    # scaling in cm:
    objp = objp * world_scaling
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    ######### Images need a white/black inverted background ##############
    image_folder = Path(folder_name)
    images = [
        str(Path(image_folder, file.name)) for file in image_folder.glob(f"*{cam}*.png")
    ]

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray_not_inv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = np.array(255 - gray_not_inv, dtype="uint8")

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chess_grid, flags=None)  # 7,6 #
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, chess_grid, corners2, ret)
            if show_output:
                cv2.imshow(f"found corners in img {fname}", img)
                cv2.waitKey(500)
        else:
            if show_output:
                cv2.imshow(f"No corners found in inverted img {fname}", gray)
                cv2.waitKey(500)

    # # close windows when "q" or "Esc" key is pressed
    if show_output:
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

    _, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return mtx, dist


def save_calibration_matrix(folder, mtx, dist, cam):
    filename = Path.joinpath(Path(folder), f"calib_cam_{cam}.npz")
    np.savez(filename, mtx=mtx, dist=dist)


def load_calibration_matrix(folder, cam):
    filename = Path.joinpath(Path(folder), f"calib_cam_{cam}.npz")
    with np.load(filename) as X:
        mtx, dist = [X[i] for i in ("mtx", "dist")]
    return mtx, dist


if __name__ == "__main__":
    chess_grid = (5, 9)
    world_scaling = 1.0289275117277425  # cm
    cam = "left"

    folder_name = "data/imgs_cam_calib/" + cam + "/"
    mtx, dist = calibrate_camera(chess_grid, world_scaling, cam, folder_name)
    print(f"{mtx=},\n {dist=}")
    # save_calibration_matrix("data/calibration_matrices/", mtx, dist, cam)
