import logging
import itertools
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


def _reduce_cluster_to_only_one_angle_conserving_pixel_of_each_row(cluster, left=True):
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
    diff_img,
    clusters,
    which_side_overlap,
    occluded_rows_clusters,
):
    angle_of_clusters = {}
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

        pos, angle, support, _, _ = calculate_position_from_cluster_and_image(
            diff_img, cluster_reduced, weighted=False
        )
        angle_of_clusters[cluster_id] = angle
        print(f"{angle=}")
    return angle_of_clusters


def combine_clusters_based_on_the_angle(clusters, angle_of_clusters):
    combined_clusters = []
    for (cluster_id_1, cluster_1), (cluster_id_2, cluster_2) in itertools.combinations(
        enumerate(clusters), 2
    ):
        if np.isclose(
            angle_of_clusters[cluster_id_1], angle_of_clusters[cluster_id_2], atol=5
        ):
            combined_clusters.append(np.vstack([cluster_1, cluster_2]))
    return combined_clusters
