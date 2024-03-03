from pathlib import Path

import numpy as np
import cv2


def stereo_calibrate(
    mtx0,
    dist0,
    mtx1,
    dist1,
    frames_prefix_c0,
    frames_prefix_c1,
    chessgrid,
    world_scaling,
):
    """
    Perform stereo calibration using a pair of synchronized camera frames.

    Args:
        mtx0 (numpy.ndarray): Camera matrix of the first camera.
        dist0 (numpy.ndarray): Distortion coefficients of the first camera.
        mtx1 (numpy.ndarray): Camera matrix of the second camera.
        dist1 (numpy.ndarray): Distortion coefficients of the second camera.
        frames_prefix_c0 (tuple): File path and prefix for the synchronized
            frames of the first camera.
        frames_prefix_c1 (tuple): File path and prefix for the synchronized
            frames of the second camera.
        chessgrid (tuple): Number of rows and columns in the calibration
            chessboard.
        world_scaling (float): Scaling factor for the world coordinates
            of the calibration chessboard.

    Returns:
        tuple: A tuple containing the rotation matrix (R), translation
        vector (T), and root mean square error (rmse) of the stereo calibration.

    Raises:
        ValueError: If the number of frames in the first and second camera
            folders are not the same, or if no images are found in the given
            folder with the specified prefix.

    Note:
        The images can be skipped by pressing 's' when the images are displayed.
        Press space to continue to the next image.

    References:
        https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    """
    # read the synched frames
    c0_images_names = sorted(Path.glob(frames_prefix_c0[0], frames_prefix_c0[1]))
    c1_images_names = sorted(Path.glob(frames_prefix_c1[0], frames_prefix_c1[1]))

    # convert path to string
    c0_image_names = [str(imname) for imname in c0_images_names]
    c1_image_names = [str(imname) for imname in c1_images_names]

    if len(c0_image_names) != len(c1_image_names):
        raise ValueError(
            "The number of frames in the first and second camera folders must be the same."
        )
    if len(c0_image_names) == 0 or len(c1_image_names) == 0:
        raise ValueError(
            f"No images found in the given folder"
            f" {frames_prefix_c0[0]} with prefix"
            f" {frames_prefix_c0[1]} or {frames_prefix_c1[1]}."
        )

    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # unpack calibration pattern settings
    rows = chessgrid[0]
    columns = chessgrid[1]

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = cv2.imread(c0_image_names[0]).shape[1]
    height = cv2.imread(c0_image_names[0]).shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for framename0, framename1 in zip(c0_image_names, c1_image_names):
        frame0 = cv2.imread(framename0)
        frame1 = cv2.imread(framename1)
        gray1 = np.array(255 - cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        gray2 = np.array(255 - cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0, 0].astype(np.int32)
            p0_c2 = corners2[0, 0].astype(np.int32)

            cv2.putText(
                frame0,
                "O",
                (p0_c1[0], p0_c1[1]),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv2.drawChessboardCorners(frame0, (rows, columns), corners1, c_ret1)
            cv2.imshow(framename0, frame0)

            cv2.putText(
                frame1,
                "O",
                (p0_c2[0], p0_c2[1]),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv2.drawChessboardCorners(frame1, (rows, columns), corners2, c_ret2)
            cv2.imshow(framename1, frame1)
            k = cv2.waitKey(0)

            # skip images if they dont allign
            if k & 0xFF == ord("s"):
                print("skipping")
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    rmse, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx0,
        dist0,
        mtx1,
        dist1,
        (width, height),
        criteria=criteria,
        flags=stereocalibration_flags,
    )

    cv2.destroyAllWindows()

    return R, T, rmse


def save_steroecalibration(folder, R, T):
    filename = Path.joinpath(Path(folder), f"stereo_RT_cl_cr.npz")
    np.savez(filename, R=R, T=T)


if __name__ == "__main__":
    from dartdetect.calibration.calibrationintrinsic import load_calibration_matrix

    calibration_folder = "data/calibration_matrices/"
    image_folder = "data/imgs_stereo_calib/"

    cam0 = "left"
    cam1 = "right"

    mtx0, dist0 = load_calibration_matrix(calibration_folder, cam0)
    mtx1, dist1 = load_calibration_matrix(calibration_folder, cam1)

    # calibration pattern settings
    chessgrid = (5, 9)
    world_scaling = 1.0289275117277425

    frame_folder_and_prefix_c0 = Path(image_folder), f"*{cam0}*"
    frame_folder_and_prefix_c1 = Path(image_folder), f"*{cam1}*"

    R, T, rmse = stereo_calibrate(
        mtx0,
        dist0,
        mtx1,
        dist1,
        frame_folder_and_prefix_c0,
        frame_folder_and_prefix_c1,
        chessgrid,
        world_scaling,
    )
    print(f"{R=},\n{T=},\n{rmse=}")

    # save_steroecalibration(calibration_folder, R, T)
