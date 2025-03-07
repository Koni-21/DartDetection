import logging
from typing import Dict, List, Tuple, Set, Any, Optional, Union, TypedDict

import numpy as np
from numpy.typing import NDArray

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


def dilate_cluster(
    cluster_mask: NDArray[np.int_], img_width: int, dilate_cluster_by_n_px: int = 1
) -> NDArray[np.int_]:
    """
    Dilates the given cluster mask with new columns by a specified
    number of pixels.

    Args:
        cluster_mask: A 2D array where each row represents a coordinate (row, col)
        img_width: The width of the image
        dilate_cluster_by_n_px: Number of pixels to dilate the cluster by

    Returns:
        The dilated cluster mask
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


def overlap(
    cluster: NDArray[np.int_], saved_dart_cluster: NDArray[np.int_]
) -> NDArray[np.int_]:
    """
    Identify overlapping points between two clusters.

    Args:
        cluster: The first cluster of points
        saved_dart_cluster: The second cluster of points

    Returns:
        Array of points present in both clusters
    """
    cluster_set: Set[Tuple[int, int]] = set(map(tuple, cluster))
    saved_dart_set: Set[Tuple[int, int]] = set(map(tuple, saved_dart_cluster))
    overlapping_points = np.array(list(cluster_set & saved_dart_set))
    return overlapping_points


class OverlapResult(TypedDict):
    """Type definition for the result of differentiate_overlap function."""

    fully_usable_rows: List[int]
    middle_occluded_rows: List[int]
    left_side_overlap_rows: List[int]
    right_side_overlap_rows: List[int]
    single_pixel_thick_overlap_rows: List[int]


def differentiate_overlap(
    cluster: NDArray[np.int_], overlap_points: NDArray[np.int_]
) -> OverlapResult:
    """
    Analyzes the overlap between a cluster and overlap points,
    categorizing rows by overlap type.

    Args:
        cluster: A 2D array of points with coordinates (row, column)
        overlap_points: A 2D array of points considered as overlap

    Returns:
        Dictionary with categorized rows based on overlap type
    """
    usable_rows: List[int] = []
    left_side_overlap_rows: List[int] = []
    right_side_overlap_rows: List[int] = []
    middle_overlap_rows: List[int] = []
    single_pixel_thick_overlap_rows: List[int] = []

    for row in np.unique(cluster[:, 0]):
        row_cluster = cluster[cluster[:, 0] == row]
        row_overlap = overlap_points[overlap_points[:, 0] == row]

        # If row has no overlap points
        if row not in overlap_points[:, 0]:
            usable_rows.append(row)
        # If row is single pixel thick and that pixel overlaps
        elif (
            len(np.unique(row_cluster[:, 1])) == 1
            and np.unique(row_cluster[:, 1])[0] in row_overlap[:, 1]
        ):
            single_pixel_thick_overlap_rows.append(row)
        # If leftmost pixel overlaps
        elif np.min(row_cluster[:, 1]) in row_overlap[:, 1]:
            left_side_overlap_rows.append(row)
        # If rightmost pixel overlaps
        elif np.max(row_cluster[:, 1]) in row_overlap[:, 1]:
            right_side_overlap_rows.append(row)
        # If middle pixels overlap
        else:
            middle_overlap_rows.append(row)

    return {
        "fully_usable_rows": usable_rows,
        "middle_occluded_rows": middle_overlap_rows,
        "single_pixel_thick_overlap_rows": single_pixel_thick_overlap_rows,
        "left_side_overlap_rows": left_side_overlap_rows,
        "right_side_overlap_rows": right_side_overlap_rows,
    }


def occlusion_kind(occluded_rows: OverlapResult, thresh_needed_rows: int = 2) -> str:
    """
    Determines the kind of occlusion based on the provided occluded rows.

    Args:
        occluded_rows: Dictionary containing categorized rows by overlap type
        thresh_needed_rows: Threshold number of rows needed to consider them usable

    Returns:
        String indicating the type of occlusion
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
        return "undefined_overlap_case"


def check_overlap(
    cluster_in: NDArray[np.int_],
    saved_darts: Dict[str, Dict[str, Any]],
    thresh_overlapping_points: int = 1,
) -> Tuple[List[int], List[NDArray[np.int_]]]:
    """
    Check for overlap between current dart cluster and previously saved darts.

    Args:
        cluster_in: The current dart cluster to check for overlap
        saved_darts: Dictionary of previously saved darts with their clusters
        thresh_overlapping_points: Minimum overlapping points to count as overlapping

    Returns:
        Tuple of overlapping dart indices and overlap points
    """
    if len(saved_darts) == 0:
        return [], []
    overlapping_darts: List[int] = []
    overlap_points: List[NDArray[np.int_]] = []

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


def check_occlusion_type_of_a_single_cluster(
    cluster_in: NDArray[np.int_], saved_darts: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Determines if and how a detected cluster is occluded by previously detected darts.

    Args:
        cluster_in: The cluster points to check for occlusion
        saved_darts: Dictionary containing previously detected darts

    Returns:
        Dictionary with occlusion details if detected, empty dict otherwise
    """
    overlapping_darts, overlap_points_list = check_overlap(cluster_in, saved_darts)
    if len(overlapping_darts) > 0:
        overlap_points = np.vstack(overlap_points_list)
        occluded_rows = differentiate_overlap(cluster_in, overlap_points)
        occlusion_dict = {
            "overlapping_darts": overlapping_darts,
            "occlusion_kind": occlusion_kind(occluded_rows),
            "overlap_points": overlap_points,
        }
        return {**occlusion_dict, **occluded_rows}
    else:
        return {}


def calculate_position_from_occluded_dart(
    occlusion_dict: Dict[str, Any],
    cluster_in: NDArray[np.int_],
    diff_img: NDArray[np.float64],
    current_img: NDArray[np.float64],
    saved_darts: Dict[str, Dict[str, Any]],
    min_usable_columns_middle_overlap: int = 1,
) -> Tuple[float, float, int, float, float, NDArray[np.int_]]:
    """
    Calculate the position of a dart from occluded dart data.

    Args:
        occlusion_dict: Dictionary containing occlusion information
        cluster_in: Cluster data for the current dart
        diff_img: Difference image used for position calculation
        current_img: Current image frame
        saved_darts: Dictionary containing saved dart data
        min_usable_columns_middle_overlap: Minimum usable columns for middle overlap

    Returns:
        Tuple of position, angle, support pixels count, correlation, error, and cluster
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


def filter_cluster_by_usable_rows(
    usable_rows: List[int], cluster: NDArray[np.int_]
) -> NDArray[np.int_]:
    """
    Filters the given cluster to include only rows specified as usable.

    Args:
        usable_rows: List of row indices that are fully usable
        cluster: A 2D array where the first column contains row indices

    Returns:
        Filtered cluster containing only the usable rows
    """
    mask = np.isin(cluster[:, 0], usable_rows)
    return cluster[mask]


def filter_middle_overlap_combined_cluster(
    middle_occluded_rows: List[int],
    overlap_points: NDArray[np.int_],
    combined_cluster: NDArray[np.int_],
    min_cols: int = 1,
) -> NDArray[np.int_]:
    """
    Filters cluster points symmetrically based on middle occluded rows and overlap points.

    Args:
        middle_occluded_rows: Row indices that are occluded in the middle
        overlap_points: Points that overlap with another dart
        combined_cluster: Combined cluster points
        min_cols: Minimum number of columns needed to use the row

    Returns:
        Filtered combined cluster points
    """
    # Filter to only keep the middle occluded rows
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


def check_which_sides_are_occluded_of_the_clusters(
    clusters: List[NDArray[np.int_]], overlap: NDArray[np.int_]
) -> Tuple[Dict[int, str], Dict[int, OverlapResult]]:
    """
    Determines which sides of the clusters are occluded based on the given overlap.

    Args:
        clusters: List of clusters where each cluster is a collection of points
        overlap: The overlap data used to determine occlusions

    Returns:
        Tuple of dictionaries with cluster ID as keys and occlusion information as values
    """
    which_side_overlap: Dict[int, str] = {}
    occluded_rows_clusters: Dict[int, OverlapResult] = {}

    for cluster_id, cluster in enumerate(clusters):
        occluded_rows = differentiate_overlap(cluster, overlap)
        side_overlap = occlusion_kind(occluded_rows)
        which_side_overlap[cluster_id] = side_overlap
        occluded_rows_clusters[cluster_id] = occluded_rows

    return which_side_overlap, occluded_rows_clusters
