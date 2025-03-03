import logging

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
    pos, angle, support, r, error = calculate_position_from_cluster_and_image(
        np.abs(diff_img - 2), cluster_out
    )
    LOGGER.info(f"Dart left: {pos=:.4f}, {angle=:.4f}, {support=}, {r=:.4f}")
    removed_dart_nr = None
    for dart_nr, values in saved_darts.items():
        if values["pos"] - tolerance_px < pos < values["pos"] + tolerance_px:
            LOGGER.info(f"Dart {dart_nr} removed.")
            removed_dart_nr = dart_nr
    return removed_dart_nr
