import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


## simulation functions
def draw_dart_subpixel(img, x_pos, angle, width):
    """
    Generate an image with a black bar of adjustable position, angle, and thickness.
    The image is represented as a NumPy array with subpixel accuracy achieved via
    grayscale gradients.

    Args:
        img (np.array): input image.
        x_pos (float): Horizontal position of the bar's center (subpixel accuracy).
        angle (float): Angle of the bar in degrees (clockwise).
        width (int): Thickness of the bar in pixels.
        array_shape (tuple): Shape of the array (height, width).

    Returns:
        np.ndarray: The generated image as a 2D array.
    """

    height, width_img = img.shape
    # Convert angle to radians
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Loop through each pixel and compute its distance to the bar
    for y in range(height):
        for x in range(width_img):
            # Rotate pixel coordinates
            x_rot = (x - x_pos) * cos_angle + (y - height + 1) * sin_angle

            distance = abs(x_rot)  # Perpendicular distance to the center of the bar

            # Apply intensity gradient based on distance
            if distance <= width / 2:
                img[y, x] = 0  # Fully black inside the bar
            elif distance <= width / 2 + 1:  # Transition zone for subpixel accuracy
                gradient_val = distance - width / 2  # Linear gradient [0, 1]
                img[y, x] = (
                    np.clip(gradient_val + img[y, x], 1, 2) - 1
                )  # if already gray value: sum up

    return img


def generate_test_images(
    img=np.ones([5, 20]),
    positions=[5, 10, 15],
    angles=[3, 2, 9],
    widths=[2, 2, 3],
    move_darts=[0, 0, 1],
):
    imgs = []

    imgs.append(img.copy())

    for i in range(0, len(positions)):
        imgs.append(img.copy())

        if i >= 1:
            if move_darts[i] == 1:
                img[:, :] = 1
                if i > 1:
                    if move_darts[i - 1] == 0:
                        for j in range(0, i - 1):
                            img = draw_dart_subpixel(
                                img.copy(), positions[j], angles[j], widths[j]
                            )

        img = draw_dart_subpixel(img.copy(), positions[i], angles[i], widths[i])
        imgs.append(img.copy())
        imgs.append(img.copy())
    return imgs


def add_noise(img, noise):
    img -= np.random.random_sample(np.shape(img)) * noise
    return np.clip(img, 0, 1)


# functions to use


def filter_noise(img, thresh):
    """
    Filters noise from the input image based on a threshold.

    Args:
        img (numpy.ndarray): The input image to be filtered.
        thesh (float): The threshold value for filtering noise. Values in the image
                    below (1 - thesh) will be kept, and values above or equal to
                    (1 - thesh) will be set to 1.

    Returns:
        numpy.ndarray: The filtered image with noise reduced.
    """
    return np.where(img < (1 - thresh), img, 1)


def compare_imgs(previous_img, current_img):
    """
    Compare two images by subtracting the previous image from the
    current image and adding 1. This way a incoming dart (0 < values < 1)
    and a removed dart (1 < values < 2) can be detected.

    Args:
        previous_img (numpy.ndarray): The previous image.
        current_img (numpy.ndarray): The current image.

    Returns:
        numpy.ndarray: The result of the comparison.
    """
    return current_img - previous_img + 1


def get_roi_coords(diff_img, incoming=True, thresh_binarise=0.1):
    """
    Get the coordinates of incoming or leaving darts/ pixels in the given image.

    This function identifies the coordinates of the pixels in the image that are
    darker (incoming) or brighter (leaving) than a specified threshold.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        thresh_binarise (float, optional): The threshold for binarizing the image.
            Pixels with values less than (1 - thresh_binarise) are considered
            part of the dart (ROI). Default is 0.1.

    Returns:
        numpy.ndarray: An array of coordinates (row, column) of the pixels that are
                    part of the ROI.
    """
    if incoming:
        diff_img_in = np.clip(
            diff_img, 0, 1
        )  # only consider the new dark pixels (incoming)
        coordinates = np.column_stack(np.where(diff_img_in < 1 - thresh_binarise))
    else:
        diff_img_in = np.clip(
            diff_img - 1, 0, 1
        )  # only consider the new bight pixels (leaving)
        coordinates = np.column_stack(np.where(diff_img_in > thresh_binarise))

    return coordinates


def find_clusters(
    coordinates, thresh_n_pixels_dart=10, dbscan_eps=1, dbscan_min_samples=2
):
    """
    Apply DBSCAN clustering to a set of coordinates and filter clusters
    based on a pixel count threshold.

    Args:
        coordinates (array-like): An array of coordinate points to be clustered.
        thresh_n_pixels_dart (int, optional): The minimum number of points
            required in a cluster to be considered valid. Default is 10.
        dbscan_eps (int): See sklearn.cluster.DBSCAN. (Defaults to 1)
        dbscan_min_samples (int): See sklearn.cluster.DBSCAN. (Defaults to 1)
    Returns:
        list: A list of arrays, each containing the coordinates of a valid cluster.
    """
    # Apply DBSCAN clustering (like flood fill with nearest neighbor = 1)
    dbscan_cluster_in = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(
        coordinates
    )

    clustered_dartsegments_coords = []
    for label in np.unique(dbscan_cluster_in.labels_):
        cluster_mask = coordinates[dbscan_cluster_in.labels_ == label].astype(int)
        if len(cluster_mask) < thresh_n_pixels_dart:
            continue
        clustered_dartsegments_coords.append(cluster_mask)

    return clustered_dartsegments_coords


def try_get_clusters_in_out(
    diff_img,
    thresh_binarise=0.1,
    thresh_n_pixels_dart=2,
    dbscan_eps=1,
    dbscan_min_samples=1,
):
    """
    Attempts to get clusters of coordinates from regions of interest (ROI)
    in an image.

    Args:
        img (ndarray): The input image from which to extract ROI coordinates.
        thresh_binarise (float, optional): The threshold value for binarizing
            the image. Default is 0.1.
        thresh_n_pixels_dart (int, optional): The minimum number of pixels
            required to consider a cluster as a dart. Default is 2.
        dbscan_eps (int): See sklearn.cluster.DBSCAN. (Defaults to 1)
        dbscan_min_samples (int): See sklearn.cluster.DBSCAN. (Defaults to 1)

    Returns:
        tuple: A tuple containing two elements:
            - clusters_in (list or None): List of clusters found in the
                incoming ROI, or None if no clusters are found.
            - clusters_out (list or None): List of clusters found in the
                outgoing ROI, or None if no clusters are found.
    """
    coords_in = get_roi_coords(diff_img, incoming=True, thresh_binarise=thresh_binarise)
    coords_out = get_roi_coords(
        diff_img, incoming=False, thresh_binarise=thresh_binarise
    )

    if len(coords_in) > thresh_n_pixels_dart:
        clusters_in = find_clusters(
            coords_in, thresh_n_pixels_dart, dbscan_eps, dbscan_min_samples
        )
    else:
        clusters_in = []

    if len(coords_out) > thresh_n_pixels_dart:
        clusters_out = find_clusters(
            coords_out, thresh_n_pixels_dart, dbscan_eps, dbscan_min_samples
        )
    else:
        clusters_out = []

    return clusters_in, clusters_out


def lin_regression_on_cluster(img, cluster):
    """
    Perform linear regression on a cluster of points in an image.
    This function calculates the row-wise mean of the cluster points in the image,
    then fits a line through these points using linear regression.

    Args:
        img (numpy.ndarray): The input image as a 2D numpy array.
        cluster (numpy.ndarray): A 2D numpy array where each row represents
            a point in the cluster, with the first column being the row indices
            and the second column being the column indices of the points in
            the image.

    Returns:
        tuple: A tuple containing:
            - pos (float): The the hitpoint of the dart as x-coordinate in the
                image corrdinate system. (The y-intercept of the fitted line at the
                maximum row index.)
            - w0 (float): The y-intercept of the fitted line.
            - w1 (float): The slope of the fitted line.
            - x (list): The list of row indices used for the regression.
            - y (list): The list of mean column indices corresponding to the row indices.
    """

    row_mean = {}
    for row in np.unique(cluster[:, 0]):
        col_idx = cluster[:, 1][cluster[:, 0] == row]
        row_vals = img[row, col_idx]
        weighted_vals = 1 - row_vals
        row_mean[row] = np.sum(weighted_vals * col_idx) / np.sum(weighted_vals)

    h, w = list(row_mean.keys()), list(row_mean.values())
    # flip axis to avoid infinit slope
    x = h
    y = w

    w1 = np.cov(x, y, bias=1)[0][1] / np.var(x)
    w0 = np.mean(y) - w1 * np.mean(x)
    pos = w0 + w1 * np.max(h)

    return (pos, w0, w1, x, y)


def check_nr_of_clusters(clusters_in, clusters_out):
    """
    Checks the number of clusters in the incoming and outgoing lists and
    returns the appropriate clusters.

    Args:
        clusters_in (list): A list of incoming clusters.
        clusters_out (list): A list of outgoing clusters.
    Returns:
        tuple: A tuple containing the selected incoming cluster and
            outgoing cluster. If there are no clusters in either list,
            an empty list is returned for that position in the tuple.
            If there are multiple clusters in either list,
            a warning is logged and only the first cluster is used.
    """
    nr_clusters_in = len(clusters_in)
    nr_clusters_out = len(clusters_out)

    if nr_clusters_in == 0 and nr_clusters_out == 0:
        return ([], [])
    elif nr_clusters_out == 0:
        cluster_in_combined = np.vstack(clusters_in)
        cluster_out_combined = []
    elif nr_clusters_in == 0:
        cluster_in_combined = []
        cluster_out_combined = np.vstack(clusters_out)
    else:
        cluster_in_combined = np.vstack(clusters_in)
        cluster_out_combined = np.vstack(clusters_out)

    if nr_clusters_in > 1:
        LOGGER.warning(
            f"More than one new 'incoming' cluster found: {nr_clusters_in=}. "
            f"Using the combined cluster with size {len(cluster_in_combined)}."
        )
    if nr_clusters_out > 1:
        LOGGER.warning(
            f"More than one new 'leaving' cluster found: {nr_clusters_out=}. "
            f"Using the combined cluster with size {len(cluster_out_combined)}."
        )

    return (cluster_in_combined, cluster_out_combined)


def dart_fully_arrived(img_height, cluster_in, distance_to_bottom=1):
    """
    Determines if the dart has fully arrived based on the image height
    and cluster_in coordinates.

    (If the detected cluster_in has no coordinates near the bottom)

    Args:
        img_height (int): height of the image.
        cluster_in (numpy.ndarray): An array of coordinates representing
            the cluster of points where the dart is.

    Returns:
        bool: True if the dart has fully arrived, False otherwise.
    """
    max_row = np.max(cluster_in[:, 0])
    if max_row < (img_height - distance_to_bottom):
        LOGGER.info(f"Dart has not arrived yet: {max_row=}, {img_height=}")
        return False
    return True


def dart_moved(diff_img, cluster_in, cluster_out, difference_thresh=1):
    """
    Determines if a dart has moved based on the difference in the sum of
    grayscale values of 'leaving' and 'incoming' pixels.

    Args:
        diff_img (np.ndarray): The difference image, where each pixel value
            represents the difference in grayscale values between two frames.
        cluster_in (np.ndarray): Array of coordinates (row, col) representing
            the 'incoming' pixels.
        cluster_out (np.ndarray): Array of coordinates (row, col) representing
            the 'leaving' pixels.
        difference_thresh (float, optional): The threshold for the difference
            in the sum of grayscale values to determine if the dart has moved.
            Defaults to 1.
    Returns:
        bool: True if the dart has moved based on the difference in the sum of
        grayscale values, False otherwise.
    """
    if cluster_out is not None:
        # plt.scatter(cluster_out[:, 1], cluster_out[:, 0], c="g", marker="x")
        out_sum = np.sum(diff_img[cluster_out[:, 0], cluster_out[:, 1]] - 1)
        in_sum = np.sum(diff_img[cluster_in[:, 0], cluster_in[:, 1]])

        if np.abs(out_sum - in_sum) <= difference_thresh:
            LOGGER.info(
                "Dart Moved based on the difference in the sum of the "
                "grayscale values of 'leaving' and 'incoming' pixel: "
                f"{out_sum=:.2f}, {in_sum=:.2f}"
            )
            return True
    return False


def overlap(cluster, saved_dart_cluster):
    """
    Identify overlapping points between two clusters.

    This function takes two clusters of points and returns an array of points
    that are present in both clusters.

    Args:
        cluster (list or array-like): The first cluster of points.
        saved_dart_cluster (list or array-like): The second cluster of points.

    Returns:
        numpy.ndarray: An array of points that are present in both clusters.
    """
    cluster_set = set(map(tuple, cluster))
    saved_dart_set = set(map(tuple, saved_dart_cluster))
    overlapping_points = np.array(list(cluster_set & saved_dart_set))
    return overlapping_points


def differentiate_overlap(cluster, overlap_points):
    """
    Analyzes the overlap between a cluster of points and overlap points,
    categorizing rows based on the type of overlap.

    Args:
        cluster (numpy.ndarray): A 2D array where each row represents a point with
                                its coordinates (row, column).
        overlap_points (numpy.ndarray): A 2D array where each row represents a point
                                        that is considered an overlap with its
                                        coordinates (row, column).
    Returns:
        dict: A dictionary with the following keys:
            - "fully_usable_rows" (list): Rows that do not have any overlap.
            - "middle_occluded_rows" (list): Rows that have overlap in the middle.
            - "left_side_overlap" (int): Number of rows with overlap on the left side.
            - "right_side_overlap" (int): Number of rows with overlap on the right side.
    """
    usable_rows = []
    left_side_overlap = 0
    right_side_overlap = 0
    middle_overlap_rows = []
    for row in np.unique(cluster[:, 0]):
        if row not in overlap_points[:, 0]:
            usable_rows.append(row)
        elif (
            np.min(cluster[cluster[:, 0] == row, 1])
            in overlap_points[overlap_points[:, 0] == row, 1]
        ):
            left_side_overlap += 1
        elif (
            np.max(cluster[cluster[:, 0] == row, 1])
            in overlap_points[overlap_points[:, 0] == row, 1]
        ):
            right_side_overlap += 1
        else:
            middle_overlap_rows.append(row)

    return {
        "fully_usable_rows": usable_rows,
        "middle_occluded_rows": middle_overlap_rows,
        "left_side_overlap": left_side_overlap,
        "right_side_overlap": right_side_overlap,
    }


def filter_cluster_by_usable_rows(usable_rows, cluster):
    """
    Filters the given cluster to include only the rows specified in usable_rows.

    Args:
        usable_rows (list or array-like): A list or array of row indices that are fully usable.
        cluster (numpy.ndarray): A 2D numpy array where the first column contains row indices.

    Returns:
        numpy.ndarray: A 2D numpy array containing only the rows from the cluster that are specified in usable_rows.
    """
    reduced_cluster = []
    for row in usable_rows:
        reduced_cluster.append(cluster[cluster[:, 0] == row])
    cluster = np.vstack(reduced_cluster)
    return cluster


def filter_middle_overlap_combined_cluster(
    middle_occluded_rows, overlap_points, combined_cluster
):
    """
    Filters the combined cluster points symmetrically on each side based ´
    on the middle occluded rows and overlap points.

    Args:
        middle_occluded_rows (ndarray): Array of row indices that are occluded
            in the middle.
        overlap_points (ndarray): Array of points that overlap with another
            dart, where each point is represented as [row, col].
        combined_cluster (ndarray): Array of combined cluster points, where
            each point is represented as [row, col].
    Returns:
        ndarray: Filtered combined cluster points.
    """
    combined_cluster = combined_cluster[
        np.isin(combined_cluster[:, 0], middle_occluded_rows)
    ]
    for row in middle_occluded_rows:
        overlapping_columns = overlap_points[overlap_points[:, 0] == row][:, 1]
        cluster_columns = combined_cluster[combined_cluster[:, 0] == row][:, 1]

        nr_cols_left = min(overlapping_columns) - min(cluster_columns)
        nr_cols_right = max(cluster_columns) - max(overlapping_columns)

        if nr_cols_left < nr_cols_right:
            combined_cluster = combined_cluster[
                ~(
                    (combined_cluster[:, 0] == row)
                    & (combined_cluster[:, 1] <= max(cluster_columns) - nr_cols_left)
                    & (combined_cluster[:, 1] >= min(cluster_columns) + nr_cols_left)
                )
            ]
        else:
            combined_cluster = combined_cluster[
                ~(
                    (combined_cluster[:, 0] == row)
                    & (combined_cluster[:, 1] >= min(cluster_columns) + nr_cols_right)
                    & (combined_cluster[:, 1] <= max(cluster_columns) - nr_cols_right)
                )
            ]
    return combined_cluster


def calculate_position_from_cluster_and_image(img, cluster):
    """
    Calculate the position, angle, support, and correlation coefficient
    from a given image and cluster.

    Args:
        img (numpy.ndarray): The image data.
        cluster (list): A list of points representing the cluster.

    Returns:
        tuple: A tuple containing:
            - pos (tuple): The calculated position.
            - angle_pred (float): The predicted angle in degrees.
            - support (int): The number of points in the cluster.
            - r (float): The correlation coefficient between x and y
                coordinates of the cluster points.
    """
    pos, b, m, x, y = lin_regression_on_cluster(img, cluster)
    angle_pred = np.degrees(np.arctan(-1 * m))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.corrcoef(x, y)[0][1]
    support = len(cluster)
    return pos, angle_pred, support, r


def occlusion_kind(occluded_rows, thresh_needed_rows=2):
    """
    Determines the kind of occlusion based on the provided occluded rows.

    Args:
        occluded_rows (dict): A dictionary containing the following keys:
            - "fully_usable_rows": List of rows that are fully usable.
            - "middle_occluded_rows": List of rows where the middle is occluded.
            - "left_side_overlap": List of rows where the left side is occluded.
            - "right_side_overlap": List of rows where the right side is occluded.
        thresh_needed_rows (int, optional): The threshold number of rows needed
            to consider them usable. Defaults to 2.
    Returns:
        str: A string indicating the type of occlusion:
            - "fully_useable" if there are more than `thresh_needed_rows` fully usable rows.
            - "one_side_fully_occluded" if one side of the dart is fully occluded.
        Raises:
            NotImplementedError: If only the center of the dart is occluded.
    """
    usable_rows = occluded_rows["fully_usable_rows"]
    middle_overlap_rows = occluded_rows["middle_occluded_rows"]
    left_side_overlap = occluded_rows["left_side_overlap"]
    right_side_overlap = occluded_rows["right_side_overlap"]

    if len(usable_rows) > thresh_needed_rows:
        LOGGER.info(
            f"Fully visible rows found: {len(usable_rows)=}, {usable_rows=}, {thresh_needed_rows=}"
        )
        return "fully_useable"
    elif len(middle_overlap_rows) > thresh_needed_rows:
        LOGGER.info(
            f"Only the center of the dart is occluded: {len(middle_overlap_rows)=}, {middle_overlap_rows=}"
        )
        return "middle_occluded"
    else:
        LOGGER.info(
            f"One side of the dart is fully occluded: {left_side_overlap=}, {right_side_overlap=}"
        )
        return "one_side_fully_occluded"


def check_overlap(cluster_in, saved_darts, thresh_overlapping_points=1):
    """
    Check for overlap between the current dart cluster and previously saved darts.

    Args:
        cluster_in (dict): The current dart cluster to check for overlap.
        saved_darts (dict): A dictionary of previously saved darts with their clusters.
        thresh_overlapping_points (int): minimum overlapping points to count as
            overlapping darts. Defaults to 1.

    Returns:
        dict: A dictionary containing information about the overlap if any is found.
            The dictionary includes:
            - "overlapping_darts" (list): Indices of the darts that overlap with the
                current dart.
            - "occlusion_kind" (str): The kind of occlusion detected.
            - "overlap_points" (numpy.ndarray): Points where the overlap occurs.
            - Additional keys from the occluded_rows dictionary.
    """
    if len(saved_darts) >= 1:
        overlapping_darts = []
        overlap_points = []

        for overlapping_dart in range(1, len(saved_darts) + 1):

            saved_dart_i = saved_darts[f"d{overlapping_dart}"]["cluster"]
            overlap_points_single = overlap(cluster_in, saved_dart_i)
            if len(overlap_points_single) >= thresh_overlapping_points:
                overlap_points.append(overlap_points_single)
                LOGGER.info(
                    f"Current dart {len(saved_darts) + 1} is overlapping"
                    f" with dart {overlapping_dart}."
                )
                overlapping_darts.append(overlapping_dart)

        if len(overlapping_darts) > 0:
            overlap_points = np.vstack(overlap_points)
            occluded_rows = differentiate_overlap(cluster_in, overlap_points)
            occlusion_dict = {
                "overlapping_darts": overlapping_darts,
                "occlusion_kind": occlusion_kind(occluded_rows),
                "overlap_points": overlap_points,
            }
            return occlusion_dict | occluded_rows
    return {}


def calculate_position_from_occluded_dart(
    occlusion_dict, cluster_in, diff_img, current_img, saved_darts
):
    """
    Calculate the position of a dart from occluded dart data.
    This function determines the position, angle, support, and Pearson
    correlation of a dart based on occlusion information and image data.
    It handles different types of occlusions and combines data from
    overlapping darts if necessary.

    Args:
        occlusion_dict (dict): Dictionary containing occlusion information.
        cluster_in (ndarray): Cluster data for the current dart.
        diff_img (ndarray): Difference image used for position calculation.
        current_img (ndarray): Current image frame.
        saved_darts (dict): Dictionary containing saved dart data.

    Returns:
        tuple: A tuple containing:
            - pos (float): Calculated position of the dart.
            - angle (float): Calculated angle of the dart.
            - support (float): nr of support pixels for the calculation.
            - r (float): Pearson correlation of the averaged rows.
            - error (float): Error value between combined dart positions.
    """
    if occlusion_dict.get("occlusion_kind", None) == "fully_useable":
        cluster_in = filter_cluster_by_usable_rows(
            occlusion_dict["fully_usable_rows"], cluster_in
        )
    elif occlusion_dict.get("occlusion_kind", None) == "middle_occluded":
        cluster_in = filter_middle_overlap_combined_cluster(
            occlusion_dict["middle_occluded_rows"],
            occlusion_dict["overlap_points"],
            cluster_in,
        )

    pos, angle, support, r = calculate_position_from_cluster_and_image(
        diff_img, cluster_in
    )
    error = 0

    if occlusion_dict.get("occlusion_kind", None) == "one_side_fully_occluded":
        overlapping_dart = np.min(occlusion_dict["overlapping_darts"])
        cluster_combined = np.vstack(
            [saved_darts[f"d{overlapping_dart}"]["cluster"], cluster_in]
        )
        diff_img_dx = compare_imgs(
            saved_darts[f"d{overlapping_dart}"]["img_pre"],
            current_img,
        )

        pos_2, angle_2, support_2, r_2 = calculate_position_from_cluster_and_image(
            diff_img_dx, cluster_combined
        )
        pos = (pos + pos_2) / 2
        angle = (angle + angle_2) / 2
        r = (r + r_2) / 2
        support = min(support, support_2)
        error = abs(pos - pos_2) / 2

    return (pos, angle, support, r, error)


def single_dart_removed(diff_img, cluster_out, saved_darts, tolerance_px=1):
    """
    Determine if a dart has been removed based on the difference image and cluster output.

    Args:
        diff_img (np.ndarray): The difference image used to detect changes.
        cluster_out (Any): The detected "leaving" cluster.
        saved_darts (dict): A dictionary of saved darts with their positions and other attributes.
        tolerance_px (int, optional): The tolerance in pixels for detecting dart removal. Defaults to 1.

    Returns:
        int or None: The number of the removed dart if detected, otherwise None.
    """
    pos, angle, support, r = calculate_position_from_cluster_and_image(
        np.abs(diff_img - 2), cluster_out
    )
    LOGGER.info(f"Dart left: {pos=:.4f}, {angle=:.4f}, {support=}, {r=:.4f}")
    removed_dart_nr = None
    for dart_nr, values in saved_darts.items():
        if values["pos"] - tolerance_px < pos < values["pos"] + tolerance_px:
            LOGGER.info(f"Dart {dart_nr} removed.")
            removed_dart_nr = dart_nr
    return removed_dart_nr


class SingleCamLocalize:
    """
    A class to localize darts using a single camera.

    Attributes:
    -----------
    image_count : int
        The number of images processed.
    imgs : list
        A list to store the last 3 imgs.
    current_img : ndarray
        The current image being processed.
    saved_darts : dict
        A dictionary to store information about detected darts.
    distance : int
        The distance between images to compare.
    dart : int
        The count of detected darts.

    Methods:
    --------
    new_image(img):
        Adds a new image and processes it if enough images are available.
    analyse_imgs():
        Analyzes the images to detect incoming and leaving darts.
    incoming_cluster_detected(diff_img, cluster_in):
        Processes the detected incoming dart cluster.
    leaving_cluster_detected(diff_img, cluster_out):
        Processes the detected leaving dart cluster.
    incoming_and_leaving_cluster_detected(diff_img, cluster_in, cluster_out):
        Processes the detected incoming and leaving dart clusters.
    visualize_stream(ax=None):
        Visualizes the current image and detected darts.
    """

    def __init__(self):
        self.image_count = 0
        self.imgs = []
        self.current_img = None

        self.saved_darts = {}
        self.distance = 1
        self.dart = 0

        # parameters of the subfunctions
        self.distance_to_bottom = 1  # for dart not yet arrived
        self.thresh_n_pixels_dart = 2  # nr of pixels needed to be a cluster
        self.dart_removed_tolerance = 1
        self.thresh_n_pixels_dart_cluster = 1
        self.thresh_binarise_cluster = 0.3
        self.dbscan_eps_cluster = 1
        self.dbscan_min_samples_cluster = 2
        self.thresh_noise = 0.1
        self.dart_moved_difference_thresh = 1

    def new_image(self, img):
        """
        Adds a new image to the list of images and updates the current image.

        This method appends the provided image to the list of images, increments the image count,
        and sets the current image to the provided image. If the image count is 2 or more, it will
        ensure that the list of images does not exceed 3 by removing the oldest image if necessary.
        If the image count is 2 or more, it will also call the `analyse_imgs` method.

        Args:
            img: The new image to be added.

        Returns:
            The result of the `analyse_imgs` method if the image count is 2 or more, otherwise None.
        """
        img = filter_noise(img, self.thresh_noise)
        self.imgs.append(img)
        self.image_count += 1
        self.current_img = img
        if self.image_count >= 2:
            if len(self.imgs) > 3:
                _ = self.imgs.pop(0)
            return self.analyse_imgs()

    def analyse_imgs(self):
        """
        Analyzes a series of images to detect incoming and leaving clusters.
        This method compares the current image with a previous image to identify
        differences and detect clusters of interest. It then determines if a
        new dart arrived or leaved and returns the new darts data.

        Returns:
            dict or None: A dictionary containing the darts information
                if any are found, otherwise None.
        """
        imgs = self.imgs

        diff_img = compare_imgs(imgs[-(self.distance + 1)], self.current_img)
        clusters_in, clusters_out = try_get_clusters_in_out(
            diff_img,
            thresh_binarise=self.thresh_binarise_cluster,
            thresh_n_pixels_dart=self.thresh_n_pixels_dart_cluster,
            dbscan_eps=self.dbscan_eps_cluster,
            dbscan_min_samples=self.dbscan_min_samples_cluster,
        )
        cluster_in, cluster_out = check_nr_of_clusters(clusters_in, clusters_out)

        if len(cluster_in) > 0 and len(cluster_out) > 0:
            return self.incoming_and_leaving_cluster_detected(
                diff_img, cluster_in, cluster_out
            )
        elif len(cluster_in) > 0:
            return self.incoming_cluster_detected(diff_img, cluster_in)
        elif len(cluster_out) > 0:  # only coordinates out
            return self.leaving_cluster_detected(diff_img, cluster_out)
        else:
            return None

    def incoming_cluster_detected(self, diff_img, cluster_in):
        """
        Detects an incoming cluster in the given difference image and processes it.
        This method checks if a dart has fully arrived or if the distance is 2.
        If so, it checks for occlusions with previously saved darts and calculates
        the position, angle, support, and radius of the dart. The results are then
        saved in the `saved_darts` dictionary with additional information.

        Args:
            diff_img (numpy.ndarray): The difference image in which the cluster
                is detected.
            cluster_in (list): The cluster data to be processed.
        Returns:
            dict: A dictionary containing the detected dart's position, angle,
                support, radius, error, cluster, and post image if a dart
                is detected and processed. Otherwise, None.
        """
        if (
            dart_fully_arrived(
                np.shape(diff_img)[0], cluster_in, self.distance_to_bottom
            )
            or self.distance == 2
        ):
            occlusion_dict = check_overlap(cluster_in, self.saved_darts)
            if len(occlusion_dict) > 0:
                pos, angle, support, r, error = calculate_position_from_occluded_dart(
                    occlusion_dict,
                    cluster_in,
                    diff_img,
                    self.current_img,
                    self.saved_darts,
                )
            else:
                pos, angle, support, r = calculate_position_from_cluster_and_image(
                    diff_img, cluster_in
                )
                error = 0

            self.distance = 1
            self.dart += 1

            self.saved_darts[f"d{self.dart}"] = {
                "pos": pos,
                "angle": angle,
                "r": r,
                "support": support,
                "error": error,
                "cluster": cluster_in,
                "img_pre": self.imgs[-self.distance],
            }

            return self.saved_darts[f"d{self.dart}"]
        else:
            self.distance = 2

    def leaving_cluster_detected(self, diff_img, cluster_out):
        """
        Detects if a dart has left the cluster and updates the saved darts accordingly.

        Args:
            diff_img (numpy.ndarray): The difference image used to detect changes.
            cluster_out (numpy.ndarray): The coordinates of the detected cluster.
        Returns:
            None
        """
        # plt.imshow(diff_img, cmap="gray")
        # plt.scatter(cluster_out[:, 1], cluster_out[:, 0], c="g", marker="x")
        removed_dart_nr = single_dart_removed(
            diff_img, cluster_out, self.saved_darts, self.dart_removed_tolerance
        )
        if removed_dart_nr is not None:
            self.saved_darts.pop(removed_dart_nr)
            self.dart -= 1

        return None

    def incoming_and_leaving_cluster_detected(self, diff_img, cluster_in, cluster_out):
        """
        Detects if a dart has moved.
        If a dart movement is detected, it returns None. Otherwise, it calls and
        returns the result of the `incoming_cluster_detected` method.

        Args:
            diff_img (numpy.ndarray): The difference image used to detect dart movement.
            cluster_in (list): The incoming cluster.
            cluster_out (list): The outgoing cluster.

        Returns:
            None or result of `incoming_cluster_detected` method: Returns None if dart
            movement is detected, otherwise returns the result of
            `incoming_cluster_detected` method.
        """
        if dart_moved(
            diff_img, cluster_in, cluster_out, self.dart_moved_difference_thresh
        ):

            return None
        else:
            return self.incoming_cluster_detected(diff_img, cluster_in)

    def visualize_stream(self, ax=None):
        """
        Visualizes the current image stream with detected darts.

        This function clears the provided axes (or creates new ones if none are provided),
        displays the current image in grayscale, and overlays the detected darts with
        different colors. Each dart's cluster points are shown as scatter points, and
        the dart's angle and position are visualized as a line. A legend is added to
        differentiate between the darts.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot the image and darts.
                If None, a new figure and axes will be created.

        """
        if ax == None:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax.clear()
        ax.imshow(self.current_img, cmap="gray")

        colors = [
            "",
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "brown",
            "orange",
            "pink",
        ]
        for dart, dart_dict in self.saved_darts.items():
            cluster_in = dart_dict["cluster"]
            diff_img = dart_dict["img_pre"]
            angle = dart_dict["angle"]
            pos = dart_dict["pos"]

            ax.scatter(
                cluster_in[:, 1],
                cluster_in[:, 0],
                c=colors[int(dart[-1])],
                s=0.8,
                alpha=0.1,
            )
            y = np.arange(0, np.shape(diff_img)[0])
            x = (
                np.tan(np.radians(angle * -1)) * y
                + pos
                - np.tan(np.radians(angle * -1)) * max(y)
            )
            ax.plot(
                x,
                y,
                color=colors[int(dart[-1])],
                linewidth=2,
                alpha=1,
                label=dart,
            )
        ax.legend()


if __name__ == "__main__":
    height, width = 5, 20
    test_img = np.ones([height, width])

    imgs = generate_test_images(
        test_img,
        positions=[5, 5, 1, 10, 18],
        angles=[7, 25, 2, -10, 0],
        widths=[2, 2, 2, 4, 2],
        move_darts=[0, 1, 0, 0, 0],
    )

    # imgs[5][:, :] = 1  # delete dart

    Loc = SingleCamLocalize()

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(0, len(imgs)):
        print(f"Processing image {i}:")

        found_dart = Loc.new_image(imgs[i])
        Loc.visualize_stream(ax)
        if found_dart is not None:
            print("Detected new Dart:")
            relevant_outputs = ["pos", "angle", "r", "support", "error"]
            for key in relevant_outputs:
                print(f"    {key:10s}: {found_dart[key]:.3f}")

        plt.pause(1)

    plt.pause(10)
