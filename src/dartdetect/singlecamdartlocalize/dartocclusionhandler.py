import logging
import numpy as np

from dartdetect.singlecamdartlocalize.dartclusteranalysis import (
    calculate_position_from_cluster_and_image,
    compare_imgs,
)

# Configure logging with timestamp, level and module information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)


def dilate_cluster(cluster_mask, img_width, dilate_cluster_by_n_px=1):
    """
    Dilates the given cluster mask with new columns by a specified number
    of pixels.

    Args:
        cluster_mask (numpy.ndarray): A 2D array where each row represents
            a coordinate (row, col) of the cluster.
        img_width (int): The width of the image.
        dilate_cluster_by_n_px (int, optional): The number of pixels to
            dilate the cluster by. Default is 1.

    Returns:
        numpy.ndarray: The dilated cluster mask.
    """
    rows, cols = cluster_mask[:, 0], cluster_mask[:, 1]
    for dilate in range(1, dilate_cluster_by_n_px + 1):
        new_cols_left = cols - dilate
        new_cols_right = cols + dilate
        valid_left = new_cols_left >= 0
        valid_right = new_cols_right < img_width
        cluster_mask = np.vstack(
            [
                cluster_mask,
                np.column_stack([rows[valid_left], new_cols_left[valid_left]]),
                np.column_stack([rows[valid_right], new_cols_right[valid_right]]),
            ]
        )
    return cluster_mask[np.lexsort((cluster_mask[:, 1], cluster_mask[:, 0]))]


### Occlusion Cases ####


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
    left_side_overlap_rows = []
    right_side_overlap_rows = []
    middle_overlap_rows = []
    single_pixel_thick_overlap_rows = []

    for row in np.unique(cluster[:, 0]):
        if row not in overlap_points[:, 0]:
            usable_rows.append(row)
        elif (
            np.min(cluster[cluster[:, 0] == row, 1])
            == np.max(cluster[cluster[:, 0] == row, 1])
            and np.min(cluster[cluster[:, 0] == row, 1])
            in overlap_points[overlap_points[:, 0] == row, 1]
        ):
            single_pixel_thick_overlap_rows.append(row)

        elif (
            np.min(cluster[cluster[:, 0] == row, 1])
            in overlap_points[overlap_points[:, 0] == row, 1]
        ):
            left_side_overlap_rows.append(row)
        elif (
            np.max(cluster[cluster[:, 0] == row, 1])
            in overlap_points[overlap_points[:, 0] == row, 1]
        ):
            right_side_overlap_rows.append(row)
        else:
            middle_overlap_rows.append(row)

    return {
        "fully_usable_rows": usable_rows,
        "middle_occluded_rows": middle_overlap_rows,
        "single_pixel_thick_overlap_rows": single_pixel_thick_overlap_rows,
        "left_side_overlap_rows": left_side_overlap_rows,
        "right_side_overlap_rows": right_side_overlap_rows,
    }


def occlusion_kind(occluded_rows, thresh_needed_rows=2):
    """
    Determines the kind of occlusion based on the provided occluded rows.

    Args:
        occluded_rows (dict): A dictionary containing the following keys:
            - "fully_usable_rows": List of rows that are fully usable.
            - "middle_occluded_rows": List of rows where the middle is occluded.
            - "left_side_overlap_rows": List of rows where the left side is occluded.
            - "right_side_overlap_rows": List of rows where the right side is occluded.
        thresh_needed_rows (int, optional): The threshold number of rows needed
            to consider them usable. Defaults to 2.
    Returns:
        str: A string indicating the type of occlusion:
            - "fully_useable" if there are more than `thresh_needed_rows` fully
                usable rows.
            - "left_side_fully_occluded" or "right_side_fully_occluded" if one
                side of the dart is fully occluded.
        Raises:
            NotImplementedError: If only the center of the dart is occluded.
    """
    usable_rows = occluded_rows["fully_usable_rows"]
    middle_overlap_rows = occluded_rows["middle_occluded_rows"]
    left_side_overlap = occluded_rows["left_side_overlap_rows"]
    right_side_overlap = occluded_rows["right_side_overlap_rows"]

    if len(usable_rows) >= thresh_needed_rows:
        LOGGER.info(
            f"Fully visible rows found: {len(usable_rows)=}, {usable_rows=}, {thresh_needed_rows=}"
        )
        return "fully_useable"
    elif len(middle_overlap_rows) > thresh_needed_rows:
        LOGGER.info(
            f"Only the center of the dart/cluster is occluded: {len(middle_overlap_rows)=}, {middle_overlap_rows=}"
        )
        return "middle_occluded"
    elif len(left_side_overlap) > len(right_side_overlap):
        LOGGER.info(
            f"Left side of the dart/cluster is fully occluded: {left_side_overlap=}, {right_side_overlap=}"
        )
        return "left_side_fully_occluded"

    elif len(right_side_overlap) > len(left_side_overlap):
        LOGGER.info(
            f"right side of the dart/cluster is fully occluded: {left_side_overlap=}, {right_side_overlap=}"
        )
        return "right_side_fully_occluded"
    else:
        raise ValueError(f"Undefined overlap case: {occluded_rows}")


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
    if len(saved_darts) == 0:
        return [], []
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

    return overlapping_darts, overlap_points


def check_occlusion_type_of_a_single_cluster(cluster_in, saved_darts):
    """
    Determines if and how a detected cluster is occluded by previously detected darts.

    This method takes a cluster (presumably a potential new dart) and checks
    if it overlaps with any previously detected darts. It then characterizes the type
    of occlusion if any exists.

    Args:
        cluster_in (list or numpy.ndarray): The cluster points to check for occlusion.
        saved_darts (dict): A dictionary containing previously detected darts.

    Returns:
        dict: If occlusion is detected, returns a dictionary containing:
            - overlapping_darts: List of previously detected darts that overlap with this cluster
            - occlusion_kind: Classification of the type of occlusion
            - overlap_points: Points where overlap occurs
            - Additional occlusion details from differentiate_overlap function
            If no occlusion is detected, returns an empty dictionary.
    """
    overlapping_darts, overlap_points = check_overlap(cluster_in, saved_darts)
    if len(overlapping_darts) > 0:
        overlap_points = np.vstack(overlap_points)
        occluded_rows = differentiate_overlap(cluster_in, overlap_points)
        occlusion_dict = {
            "overlapping_darts": overlapping_darts,
            "occlusion_kind": occlusion_kind(occluded_rows),
            "overlap_points": overlap_points,
        }
        return occlusion_dict | occluded_rows
    else:
        return {}


def calculate_position_from_occluded_dart(
    occlusion_dict,
    cluster_in,
    diff_img,
    current_img,
    saved_darts,
    min_usable_columns_middle_overlap=1,
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
    weighted = True
    if occlusion_dict.get("occlusion_kind", None) == "fully_useable":
        cluster_in = filter_cluster_by_usable_rows(
            occlusion_dict["fully_usable_rows"], cluster_in
        )
    elif occlusion_dict.get("occlusion_kind", None) == "middle_occluded":
        cluster_in = filter_middle_overlap_combined_cluster(
            occlusion_dict["middle_occluded_rows"],
            occlusion_dict["overlap_points"],
            cluster_in,
            min_cols=min_usable_columns_middle_overlap,
        )
        weighted = False

    pos, angle, support, r, error = calculate_position_from_cluster_and_image(
        diff_img, cluster_in, weighted=weighted
    )

    if occlusion_dict.get("occlusion_kind", None) in [
        "left_side_fully_occluded",
        "right_side_fully_occluded",
    ]:
        overlapping_dart = np.min(occlusion_dict["overlapping_darts"])
        cluster_combined = np.vstack(
            [saved_darts[f"d{overlapping_dart}"]["cluster"], cluster_in]
        )
        diff_img_dx = compare_imgs(
            saved_darts[f"d{overlapping_dart}"]["img_pre"],
            current_img,
        )

        pos_2, angle_2, support_2, r_2, error = (
            calculate_position_from_cluster_and_image(diff_img_dx, cluster_combined)
        )
        pos = (pos + pos_2) / 2
        angle = (angle + angle_2) / 2
        r = (r + r_2) / 2
        support = min(support, support_2)
        error_pos = abs(pos - pos_2)
        if error_pos > error:
            error = error_pos

    return (pos, angle, support, r, error, cluster_in)


#### Fully usable rows detected


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
    middle_occluded_rows, overlap_points, combined_cluster, min_cols=1
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
        min_cols (int): Minimum number of columns needed to us the row .

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
                ~((combined_cluster[:, 0] == row) & (nr_cols_left < min_cols))
            ]
            combined_cluster = combined_cluster[
                ~(
                    (combined_cluster[:, 0] == row)
                    & (combined_cluster[:, 1] <= max(cluster_columns) - nr_cols_left)
                    & (combined_cluster[:, 1] >= min(cluster_columns) + nr_cols_left)
                )
            ]
        else:
            combined_cluster = combined_cluster[
                ~((combined_cluster[:, 0] == row) & (nr_cols_right < min_cols))
            ]
            combined_cluster = combined_cluster[
                ~(
                    (combined_cluster[:, 0] == row)
                    & (combined_cluster[:, 1] >= min(cluster_columns) + nr_cols_right)
                    & (combined_cluster[:, 1] <= max(cluster_columns) - nr_cols_right)
                )
            ]
    return combined_cluster


def check_which_sides_are_occluded_of_the_clusters(clusters, overlap):
    """
    Determines which sides of the clusters are occluded based on the given overlap.

    Args:
        clusters (list): A list of clusters where each cluster is a collection
            of points or data.
        overlap (any): The overlap data used to determine occlusions.

    Returns:
        tuple: A tuple containing two dictionaries:
            - which_side_overlap (dict): A dictionary where the key is the
                cluster ID and the value is the side overlap information.
            - occluded_rows_clusters (dict): A dictionary where the key is
                the cluster ID and the value is the occluded rows information.
    """
    which_side_overlap = {}
    occluded_rows_clusters = {}
    for cluster_id, cluster in enumerate(clusters):
        occluded_rows = differentiate_overlap(cluster, overlap)
        side_overlap = occlusion_kind(occluded_rows)
        which_side_overlap[cluster_id] = side_overlap
        occluded_rows_clusters[cluster_id] = occluded_rows
    return which_side_overlap, occluded_rows_clusters
