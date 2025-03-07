import logging
from typing import Dict, Optional, Any, Union, Tuple

import numpy as np


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


def dart_fully_arrived(
    img_height: int, cluster_in: np.ndarray, distance_to_bottom: int = 1
) -> bool:
    """
    Determines if the dart has fully arrived based on the image height
    and cluster_in coordinates.

    (If the detected cluster_in has no coordinates near the bottom)

    Args:
        img_height: height of the image.
        cluster_in: An array of coordinates representing
            the cluster of points where the dart is.
        distance_to_bottom: Distance threshold from bottom. Defaults to 1.

    Returns:
        True if the dart has fully arrived, False otherwise.
    """
    max_row = np.max(cluster_in[:, 0])
    if max_row < (img_height - distance_to_bottom):
        LOGGER.info(f"Dart has not arrived yet: {max_row=}, {img_height=}")
        return False
    return True


def dart_moved(
    diff_img: np.ndarray,
    cluster_in: np.ndarray,
    cluster_out: Optional[np.ndarray],
    difference_thresh: float = 1,
) -> bool:
    """
    Determines if a dart has moved based on the difference in the sum of
    grayscale values of 'leaving' and 'incoming' pixels.

    Args:
        diff_img: The difference image, where each pixel value
            represents the difference in grayscale values between two frames.
        cluster_in: Array of coordinates (row, col) representing
            the 'incoming' pixels.
        cluster_out: Array of coordinates (row, col) representing
            the 'leaving' pixels. Can be None if no leaving pixels detected.
        difference_thresh: The threshold for the difference
            in the sum of grayscale values to determine if the dart has moved.
            Defaults to 1.
    Returns:
        True if the dart has moved based on the difference in the sum of
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


def single_dart_removed(
    diff_img: np.ndarray,
    cluster_out: np.ndarray,
    saved_darts: Dict[Any, Dict[str, float]],
    tolerance_px: int = 1,
) -> Optional[str]:
    """
    Determine if a dart has been removed based on the difference image and cluster output.

    Args:
        diff_img: The difference image used to detect changes.
        cluster_out: The detected "leaving" cluster as array of coordinates.
        saved_darts: A dictionary of saved darts with their positions
            and other attributes.
        tolerance_px: The tolerance in pixels for detecting dart removal.
            Defaults to 1.

    Returns:
        The ID number of the removed dart (e.g. 'd1') if detected, otherwise None.
    """
    # Validate input data
    if cluster_out is None or len(cluster_out) == 0:
        LOGGER.warning("No valid leaving cluster detected")
        return None

    pos, angle, support, r, error = calculate_position_from_cluster_and_image(
        np.abs(diff_img - 2), cluster_out
    )
    LOGGER.info(f"Dart left: {pos=:.4f}, {angle=:.4f}, {support=}, {r=:.4f}")

    return _find_matching_dart(saved_darts, pos, tolerance_px)


def _find_matching_dart(
    saved_darts: Dict[Any, Dict[str, float]], position: float, tolerance_px: int = 1
) -> Optional[str]:
    """
    Find a dart in saved_darts that matches the given position within tolerance.

    Args:
        saved_darts: Dictionary of saved darts with their attributes.
        position: The position to match.
        tolerance_px: Tolerance in pixels for position matching. Defaults to 1.

    Returns:
        The matching dart ID (e.g. 'd1') if found, otherwise None.
    """
    darts_removed = []
    for dart_id, values in saved_darts.items():
        if values["pos"] - tolerance_px <= position <= values["pos"] + tolerance_px:
            darts_removed.append(dart_id)

    if len(darts_removed) == 0:
        return None
    elif len(darts_removed) == 1:
        LOGGER.info(f"Dart {darts_removed[0]} removed.")
    elif len(darts_removed) > 1:
        LOGGER.warning(
            f"Multiple darts removed: {darts_removed}. "
            "This should not happen. Check the tolerance. "
            f"Only removing {darts_removed[0]}."
        )
    return darts_removed[0]
