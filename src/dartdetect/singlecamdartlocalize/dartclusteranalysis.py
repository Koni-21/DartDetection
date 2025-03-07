import numpy as np

from sklearn.cluster import DBSCAN


def filter_noise(img: np.ndarray, thresh: float) -> np.ndarray:
    """
    Filters noise from the input image based on a threshold.

    Args:
        img: The input image to be filtered.
        thresh: The threshold value for filtering noise. Values in the image
                below (1 - thresh) will be kept, and values above or equal to
                (1 - thresh) will be set to 1.

    Returns:
        The filtered image with noise reduced.
    """
    return np.where(img < (1 - thresh), img, 1)


def compare_imgs(previous_img: np.ndarray, current_img: np.ndarray) -> np.ndarray:
    """
    Compare two images by subtracting the previous image from the
    current image and adding 1. This way a incoming dart (0 < values < 1)
    and a removed dart (1 < values < 2) can be detected.

    Args:
        previous_img: The previous image.
        current_img: The current image.

    Returns:
        The result of the comparison.
    """
    return current_img - previous_img + 1


def get_roi_coords(
    diff_img: np.ndarray, incoming: bool = True, thresh_binarise: float = 0.1
) -> np.ndarray:
    """
    Get the coordinates of incoming or leaving darts/ pixels in the given image.

    This function identifies the coordinates of the pixels in the image that are
    darker (incoming) or brighter (leaving) than a specified threshold.

    Args:
        diff_img: The input image as a NumPy array.
        incoming: Whether to look for incoming or leaving darts.
        thresh_binarise: The threshold for binarising the image.
            Pixels with values less than (1 - thresh_binarise) are considered
            part of the dart (ROI). Default is 0.1.

    Returns:
        An array of coordinates (row, column) of the pixels that are
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
    coordinates: np.ndarray,
    thresh_n_pixels_dart: int = 10,
    dbscan_eps: int = 1,
    dbscan_min_samples: int = 2,
) -> list[np.ndarray]:
    """
    Apply DBSCAN clustering to a set of coordinates and filter clusters
    based on a pixel count threshold.

    Args:
        coordinates: An array of coordinate points to be clustered.
        thresh_n_pixels_dart: The minimum number of points
            required in a cluster to be considered valid.
        dbscan_eps: See sklearn.cluster.DBSCAN.
        dbscan_min_samples: See sklearn.cluster.DBSCAN.
    Returns:
        A list of arrays, each containing the coordinates of a valid cluster.
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
    diff_img: np.ndarray,
    thresh_binarise: float = 0.1,
    thresh_n_pixels_dart: int = 2,
    dbscan_eps: int = 1,
    dbscan_min_samples: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Attempts to get clusters of coordinates from regions of interest (ROI)
    in an image.

    Args:
        diff_img: The input image from which to extract ROI coordinates.
        thresh_binarise: The threshold value for binarizing the image.
        thresh_n_pixels_dart: The minimum number of pixels
            required to consider a cluster as a dart.
        dbscan_eps: See sklearn.cluster.DBSCAN.
        dbscan_min_samples: See sklearn.cluster.DBSCAN.

    Returns:
        A tuple containing two elements:
            - clusters_in: List of clusters found in the incoming ROI,
                or empty list if no clusters are found.
            - clusters_out: List of clusters found in the outgoing ROI,
                or empty list if no clusters are found.
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


def weighted_average_rows(
    img: np.ndarray, cluster: np.ndarray
) -> tuple[list[int], list[float], int]:
    """
    Calculate the weighted average column positions for each unique row in the cluster.

    Args:
        img: The input image array.
        cluster: A 2D array where each row represents a point with
                [row_index, col_index].
    Returns:
        A tuple containing:
            - height_idx: List of unique row indices from the cluster.
            - averaged_row_width_position: List of weighted average column positions
                                          corresponding to each row index.
            - error: Placeholder for error, currently always returns 0.
    """
    row_mean = {}
    for row in np.unique(cluster[:, 0]):
        col_idx = cluster[:, 1][cluster[:, 0] == row]
        row_vals = img[row, col_idx]
        weighted_vals = 1 - row_vals
        row_mean[row] = np.sum(weighted_vals * col_idx) / np.sum(weighted_vals)

    height_idx, averaged_row_width_position = list(row_mean.keys()), list(
        row_mean.values()
    )
    error = 0

    return height_idx, averaged_row_width_position, error


def average_rows(cluster: np.ndarray) -> tuple[list[int], list[float], float]:
    """
    Calculate the average column index for each unique row in the given cluster.

    Args:
        cluster: A 2D array where each row represents a point with
                the first column as the row index and the second
                column as the column index.
    Returns:
        A tuple containing:
            - height_idx: A list of unique row indices.
            - averaged_row_width_position: A list of average column indices
                                          corresponding to each unique row.
            - error: An error estimation value based on the number of unique rows.
    """
    row_mean = {}
    for row in np.unique(cluster[:, 0]):
        col_idx = cluster[:, 1][cluster[:, 0] == row]
        row_mean[row] = np.mean(col_idx)

    height_idx, averaged_row_width_position = list(row_mean.keys()), list(
        row_mean.values()
    )

    # see error estimation:
    # G:\My Drive\Dartprojekt\Software\ESP32cam\250110_statistical_error_analysis_unweighted_mean.ipynb
    error = 1 / np.sqrt(len(np.unique(cluster[:, 0])))
    if error > 0.5:
        error = 0.5

    return height_idx, averaged_row_width_position, error


def lin_regression_on_cluster(
    img: np.ndarray, cluster: np.ndarray, weighted: bool = True
) -> tuple[float, float, float, list[int], list[float], float]:
    """
    Perform linear regression on a cluster of points in an image.
    This function calculates the row-wise mean of the cluster points in the image,
    then fits a line through these points using linear regression.

    Args:
        img: The input image as a 2D numpy array.
        cluster: A 2D numpy array where each row represents
            a point in the cluster, with the first column being the row indices
            and the second column being the column indices of the points in
            the image.
        weighted: Calculate the regression weighted on image values or
            only unweighted using only the cluster.

    Returns:
        A tuple containing:
            - pos: The the hitpoint of the dart as x-coordinate in the
                image corrdinate system. (The y-intercept of the fitted line at the
                maximum row index.)
            - w0: The y-intercept of the fitted line.
            - w1: The slope of the fitted line.
            - x: The list of row indices used for the regression.
            - y: The list of mean column indices corresponding to the row indices.
            - error: Estimator of the position discretization error
    """

    if weighted:
        height_idx, averaged_row_width_position, pos_discretization_error = (
            weighted_average_rows(img, cluster)
        )
    else:
        height_idx, averaged_row_width_position, pos_discretization_error = (
            average_rows(cluster)
        )

    # flip axis to avoid infinit slope
    x = height_idx
    y = averaged_row_width_position

    # https://en.wikipedia.org/wiki/Simple_linear_regression
    w1 = np.cov(x, y, bias=1)[0][1] / np.var(x)  # m
    w0 = np.mean(y) - w1 * np.mean(x)  # b
    pos = w0 + w1 * (np.shape(img)[0] - 1)

    return (pos, w0, w1, x, y, pos_discretization_error)


def calculate_position_from_cluster_and_image(
    img: np.ndarray, cluster: np.ndarray, weighted: bool = True
) -> tuple[float, float, int, float, float]:
    """
    Calculate the position, angle, support, and correlation coefficient
    from a given image and cluster.

    Args:
        img: The image data.
        cluster: A list of points representing the cluster.
        weighted: Calculate the regression weighted on image values or
            only unweighted using only the cluster.

    Returns:
        A tuple containing:
            - pos: The calculated position.
            - angle_pred: The predicted angle in degrees.
            - support: The number of points in the cluster.
            - r: The correlation coefficient between x and y
                coordinates of the cluster points.
            - error: Estimator of the position discretization error
    """
    pos, b, m, x, y, error = lin_regression_on_cluster(img, cluster, weighted=weighted)
    angle_pred = np.degrees(np.arctan(-1 * m))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.corrcoef(x, y)[0][1]
    if np.isnan(r):
        r = 1
    support = len(np.unique(cluster[:, 0]))
    return pos, angle_pred, support, r, error
