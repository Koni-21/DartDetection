import logging
import itertools
from typing import Dict, List, Tuple, Optional, Union

import numpy as np

### Cases with multiple clusters detected
from dartdetect.singlecamdartlocalize.dartocclusionhandler import (
    differentiate_overlap,
    occlusion_kind,
    filter_cluster_by_usable_rows,
)

from dartdetect.singlecamdartlocalize.dartclusteranalysis import (
    calculate_position_from_cluster_and_image,
)

# Configure logging with timestamp, level and module information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Initialize module logger using the automatic name
LOGGER = logging.getLogger(__name__)


def _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row(
    cluster: np.ndarray, left: bool = True
) -> np.ndarray:
    """
    Reduces a cluster of points to only one point per row, preserving either the rightmost or leftmost pixel in each row.

    This function processes a cluster (represented as a numpy array of [row, col] coordinates) and
    returns a new array containing only one pixel per row - either the rightmost pixel (maximum column value)
    when left=True or the leftmost pixel (minimum column value) when left=False.

    Args:
        cluster (numpy.ndarray): A numpy array of shape (n, 2) where each row represents a pixel coordinate [row, col].
        left (bool, optional): If True, preserve the rightmost pixel of each row. If False, preserve the leftmost pixel.
            Defaults to True.

    Returns:
        numpy.ndarray: A numpy array containing one pixel per row, with shape (m, 2) where m is the number of unique rows
        in the input cluster.
    """
    if left:
        return np.array(
            [
                [row, np.max(cluster[cluster[:, 0] == row][:, 1])]
                for row in np.unique(cluster[:, 0])
            ]
        )
    else:
        return np.array(
            [
                [row, np.min(cluster[cluster[:, 0] == row][:, 1])]
                for row in np.unique(cluster[:, 0])
            ]
        )


def calculate_angle_of_different_clusters(
    diff_img: np.ndarray,
    clusters: List[np.ndarray],
    which_side_overlap: Dict[int, str],
    occluded_rows_clusters: Dict[int, List[int]],
) -> Dict[int, float]:
    """
    Calculate angles for different clusters based on occlusion conditions.
    This function processes each cluster according to its occlusion state and
    calculates an angle for it. Clusters with different occlusion patterns
    (left side, right side, or fully usable) are handled using different
    preprocessing methods before angle calculation.

    Args:
        diff_img (numpy.ndarray): Difference image used for position and angle calculation.
        clusters (List[numpy.ndarray]): List of pixel clusters to process.
        which_side_overlap (Dict[int, str]): Dictionary indicating occlusion state for each cluster
                           (e.g., "left_side_fully_occluded", "right_side_fully_occluded",
                           "fully_usable").
        occluded_rows_clusters (Dict[int, List[int]]): Information about which rows are occluded in each cluster.

    Returns:
        Dict[int, float]: Dictionary mapping cluster IDs to their calculated angles.
    """
    angle_of_clusters: Dict[int, float] = {}
    for cluster_id, cluster in enumerate(clusters):
        if which_side_overlap[cluster_id] == "right_side_fully_occluded":
            cluster_reduced = (
                _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row(
                    cluster, False
                )
            )
        elif which_side_overlap[cluster_id] == "left_side_fully_occluded":
            cluster_reduced = (
                _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row(
                    cluster, True
                )
            )
        elif which_side_overlap[cluster_id] == "fully_usable":
            cluster_reduced = filter_cluster_by_usable_rows(
                occluded_rows_clusters[cluster_id], cluster
            )
        else:
            LOGGER.warning(
                f"To calculate the angle of cluster: {cluster_id}: "
                f"{which_side_overlap[cluster_id]} "
                "the unreduced cluster is used."
            )
            cluster_reduced = cluster

        _, angle, _, _, _ = calculate_position_from_cluster_and_image(
            diff_img, cluster_reduced, weighted=False
        )
        angle_of_clusters[cluster_id] = angle
        LOGGER.debug(f"Calculated angle={angle} for cluster {cluster_id}")
    return angle_of_clusters


def combine_clusters_based_on_the_angle(
    clusters: List[np.ndarray], angle_of_clusters: Dict[int, float]
) -> List[np.ndarray]:
    """
    Combines pairs of clusters that have similar angles within a specified tolerance.

    This function compares all possible pairs of clusters and creates new combined
    clusters for pairs whose corresponding angles are close to each other (within
    a tolerance of 5 degrees). The combined clusters are formed by vertically
    stacking the points of the original clusters.

    Note: If multiple clusters have similar angles (e.g., clusters A, B, and C),
    the function will create separate combined clusters for each pair (e.g., A+B, B+C, A+C),
    rather than a single combined cluster (e.g., A+B+C).

    Args:
        clusters (List[numpy.ndarray]): List of clusters, where each cluster is a collection of points.
        angle_of_clusters (Dict[int, float]): Dictionary of angles corresponding to each cluster.

    Returns:
        List[numpy.ndarray]: A list of combined clusters, where each combined cluster is formed by
              vertically stacking the points of a pair of clusters with similar angles.
    """
    combined_clusters: List[np.ndarray] = []
    angle_tolerance: float = 5.0

    for (cluster_id_1, cluster_1), (cluster_id_2, cluster_2) in itertools.combinations(
        enumerate(clusters), 2
    ):
        if np.isclose(
            angle_of_clusters[cluster_id_1],
            angle_of_clusters[cluster_id_2],
            atol=angle_tolerance,
        ):
            combined_clusters.append(np.vstack([cluster_1, cluster_2]))
    return combined_clusters
