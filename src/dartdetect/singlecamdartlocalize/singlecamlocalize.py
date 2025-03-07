import logging
import itertools
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray

from dartdetect.singlecamdartlocalize.dartclusteranalysis import (
    calculate_position_from_cluster_and_image,
    filter_noise,
    compare_imgs,
    try_get_clusters_in_out,
)
from dartdetect.singlecamdartlocalize.dartocclusionhandler import (
    dilate_cluster,
    check_occlusion_type_of_a_single_cluster,
    calculate_position_from_occluded_dart,
    check_overlap,
    check_which_sides_are_occluded_of_the_clusters,
)
from dartdetect.singlecamdartlocalize.dartmovementutils import (
    dart_fully_arrived,
    single_dart_removed,
    dart_moved,
)
from dartdetect.singlecamdartlocalize.moveddartocclusion import (
    calculate_angle_of_different_clusters,
    combine_clusters_based_on_the_angle,
)


# Configure logging with timestamp, level and module information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Initialize module logger using the automatic name
LOGGER = logging.getLogger(__name__)


class SingleCamLocalize:
    """
    A class to localize darts using a single camera.

    This class processes a sequence of images to detect, track, and localize darts
    in the field of view of a single camera.

    Attributes:
        image_empty (NDArray[np.float64]): The initial empty image used as reference.
        image_count (int): The number of images processed.
        imgs (List[NDArray[np.float64]]): A list to store the last 3 images.
        current_img (NDArray[np.float64]): The current image being processed.
        current_diff_img (NDArray[np.float64]): The difference between current and previous image.
        saved_darts (Dict[str, Dict]): A dictionary to store information about detected darts.
        nr_imgs_distance (int): The distance between images to compare.
        dart (int): The count of detected darts.
        occlusion_dict (Dict): Dictionary to store occlusion information.
    """

    def __init__(self) -> None:
        """Initialize the SingleCamLocalize object with default parameters."""
        # Image data
        self.image_empty: Optional[NDArray[np.float64]] = None
        self.image_count: int = 0
        self.imgs: List[NDArray[np.float64]] = []
        self.current_img: Optional[NDArray[np.float64]] = None
        self.current_diff_img: Optional[NDArray[np.float64]] = None

        # Dart tracking data
        self.saved_darts: Dict[str, Dict[str, Any]] = {}
        self.nr_imgs_distance: int = 1
        self.dart: int = 0
        self.occlusion_dict: Dict[str, Any] = {}

        # Analysis parameters
        self.clusters_in: List[NDArray[np.float64]] = []
        self.clusters_out: List[NDArray[np.float64]] = []

        # Configuration parameters
        self.distance_to_bottom_not_arrived: int = 1  # for dart not yet arrived
        self.dart_removed_tolerance: int = 1
        self.thresh_n_pixels_dart_cluster: int = (
            2  # nr of pixels needed to be a cluster
        )
        self.thresh_binarise_cluster: float = 0.3
        self.dbscan_eps_cluster: int = 1
        self.dbscan_min_samples_cluster: int = 2
        self.thresh_noise: float = 0.1
        self.dart_moved_difference_thresh: int = 1
        self.dilate_cluster_by_n_px: int = 1
        self.min_usable_columns_middle_overlap: int = 1

    def new_image(self, img: NDArray[np.float64]) -> Optional[Dict[str, Any]]:
        """
        Process a new image for dart detection.

        Adds the image to the processing queue and analyzes it if enough images are available.

        Args:
            img: The new image to be processed.

        Returns:
            Optional[Dict[str, Any]]: Information about newly detected dart if found, else None.
        """
        img = filter_noise(img, self.thresh_noise)
        self.imgs.append(img)
        self.image_count += 1

        if self.image_count == 1:
            self.image_empty = img

        self.current_img = img

        if self.image_count >= 2:
            if len(self.imgs) > 3:
                _ = self.imgs.pop(0)
            return self.analyse_imgs()

        return None

    def analyse_imgs(self) -> Optional[Dict[str, Any]]:
        """
        Analyze images to detect incoming and leaving dart clusters.

        This method compares the current image with a previous image to identify
        differences and detect clusters of interest. It then determines if a
        new dart arrived or left and returns the new dart's data.

        Returns:
            Optional[Dict[str, Any]]: Information about newly detected dart if found, else None.
        """
        # Get reference image based on configured image distance
        ref_img = self.imgs[-(self.nr_imgs_distance + 1)]

        # Generate difference image and extract clusters
        diff_img = compare_imgs(ref_img, self.current_img)
        self.current_diff_img = diff_img

        clusters_in, clusters_out = self._extract_clusters(diff_img)
        self.clusters_in, self.clusters_out = clusters_in, clusters_out

        # No clusters detected
        if len(clusters_in) == 0 and len(clusters_out) == 0:
            return None

        # Single incoming cluster
        elif len(clusters_in) == 1 and len(clusters_out) == 0:
            return self.incoming_cluster_detected(diff_img, clusters_in[0])

        # Both incoming and leaving clusters
        elif len(clusters_in) == 1 and len(clusters_out) == 1:
            return self.one_incoming_and_leaving_cluster_detected(
                diff_img, clusters_in[0], clusters_out[0]
            )

        # Multiple incoming clusters
        elif len(clusters_in) > 1:
            return self.multiple_clusters_detected(diff_img, clusters_in, clusters_out)

        # Only leaving clusters
        elif len(clusters_out) >= 1:
            self.check_view_empty()
            return self.leaving_cluster_detected(diff_img, np.vstack(clusters_out))

        return None

    def _extract_clusters(
        self, diff_img: NDArray[np.float64]
    ) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
        """
        Extract incoming and outgoing clusters from the difference image.

        Args:
            diff_img: Difference image to analyze.

        Returns:
            Tuple containing lists of incoming and outgoing clusters.
        """
        return try_get_clusters_in_out(
            diff_img,
            thresh_binarise=self.thresh_binarise_cluster,
            thresh_n_pixels_dart=self.thresh_n_pixels_dart_cluster,
            dbscan_eps=self.dbscan_eps_cluster,
            dbscan_min_samples=self.dbscan_min_samples_cluster,
        )

    def check_view_empty(self) -> None:
        """
        Check if the camera view is empty and reset if needed.

        Compares the current image with the initial empty image to determine
        if all darts have been removed or if the view was initially not empty.
        """
        if self.image_empty is None or self.current_img is None:
            return None

        diff_img = compare_imgs(self.image_empty, self.current_img)
        clusters_in, clusters_out = self._extract_clusters(diff_img)

        if len(clusters_out) > 0:
            LOGGER.warning("View of the camera was not empty at the start. Restarting.")
            self.reset()
        elif len(clusters_in) == 0:
            LOGGER.info("View of the camera is empty. Restarting.")
            self.reset()

    def incoming_cluster_detected(
        self, diff_img: NDArray[np.float64], cluster_in: NDArray[np.float64]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a detected incoming dart cluster.

        Determines if a dart has fully arrived, checks for occlusions with existing darts,
        and calculates position, angle and other properties of the dart.

        Args:
            diff_img: The difference image containing the cluster.
            cluster_in: Coordinates of the detected cluster.

        Returns:
            Dict[str, Any]: Information about the detected dart.
        """
        # Check if dart has fully arrived
        img_height = np.shape(diff_img)[0]
        if (
            not dart_fully_arrived(
                img_height, cluster_in, self.distance_to_bottom_not_arrived
            )
            and self.nr_imgs_distance < 2
        ):
            self.nr_imgs_distance = 2
            return None

        self.nr_imgs_distance = 1
        cluster_mod = cluster_in.copy()

        # Process cluster for occlusion detection
        cluster_in = dilate_cluster(
            cluster_in, np.shape(diff_img)[1], self.dilate_cluster_by_n_px
        )

        # Check for occlusions with existing darts
        if not self.occlusion_dict:
            self.occlusion_dict = check_occlusion_type_of_a_single_cluster(
                cluster_in, self.saved_darts
            )

        # Calculate dart position based on occlusion status
        if self.occlusion_dict:
            pos, angle, support, r, error, cluster_mod = self._process_occluded_dart(
                diff_img, cluster_in
            )
        else:
            pos, angle, support, r, error = calculate_position_from_cluster_and_image(
                diff_img, cluster_in
            )

        # Save the new dart
        self.dart += 1
        dart_id = f"d{self.dart}"

        self.saved_darts[dart_id] = {
            "pos": pos,
            "angle": angle,
            "r": r,
            "support": support,
            "error": error,
            "cluster": cluster_in,
            "cluster_mod": cluster_mod,
            "img_pre": self.imgs[-self.nr_imgs_distance - 1],
        }

        self.occlusion_dict = {}
        return self.saved_darts[dart_id]

    def _process_occluded_dart(
        self, diff_img: NDArray[np.float64], cluster_in: NDArray[np.float64]
    ) -> Tuple[float, float, int, float, float, NDArray[np.float64]]:
        """
        Calculate position for an occluded dart.

        Args:
            diff_img: The difference image containing the cluster.
            cluster_in: Coordinates of the detected cluster.

        Returns:
            Tuple containing position, angle, support, radius, error and modified cluster.
        """
        return calculate_position_from_occluded_dart(
            self.occlusion_dict,
            cluster_in,
            diff_img,
            self.current_img,
            self.saved_darts,
            self.min_usable_columns_middle_overlap,
        )

    def multiple_clusters_detected(
        self,
        diff_img: NDArray[np.float64],
        clusters_in: List[NDArray[np.float64]],
        clusters_out: List[NDArray[np.float64]],
    ) -> Optional[Dict[str, Any]]:
        """
        Process multiple detected clusters.

        Handles the case when multiple incoming clusters are detected, which could be
        from a single dart that appears as multiple clusters due to occlusion.

        Args:
            diff_img: The difference image containing the clusters.
            clusters_in: List of incoming clusters.
            clusters_out: List of outgoing clusters.

        Returns:
            Optional[Dict[str, Any]]: Information about the detected dart if successful.
        """
        LOGGER.warning(
            f"More than one new cluster found: {len(clusters_in)=}, {len(clusters_out)=}). "
        )

        if self.nr_imgs_distance < 2:
            self.nr_imgs_distance = 2
            return None

        # Check for fully usable clusters
        fully_usable_clusters, occlusion_dicts = self._find_fully_usable_clusters(
            diff_img, clusters_in
        )

        # Handle case with one fully usable cluster
        if len(fully_usable_clusters) == 1:
            self.occlusion_dict = occlusion_dicts[fully_usable_clusters[0]]
            return self.incoming_cluster_detected(
                diff_img, clusters_in[fully_usable_clusters[0]]
            )

        # Handle case with multiple fully usable clusters (select largest)
        elif len(fully_usable_clusters) > 1:
            return self._process_multiple_usable_clusters(
                diff_img, clusters_in, fully_usable_clusters, occlusion_dicts
            )

        # Check for overlapping darts
        return self._process_overlapping_clusters(diff_img, clusters_in)

    def _find_fully_usable_clusters(
        self, diff_img: NDArray[np.float64], clusters_in: List[NDArray[np.float64]]
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Find fully usable clusters among incoming clusters.

        Args:
            diff_img: The difference image.
            clusters_in: List of incoming clusters.

        Returns:
            Tuple containing list of indices of fully usable clusters and their occlusion dicts.
        """
        fully_usable_clusters = []
        occlusion_dicts = []

        for nr, cluster_in in enumerate(clusters_in):
            dilated_cluster = dilate_cluster(
                cluster_in, np.shape(diff_img)[1], self.dilate_cluster_by_n_px
            )
            occlusion_dict = check_occlusion_type_of_a_single_cluster(
                dilated_cluster, self.saved_darts
            )

            if occlusion_dict["occlusion_kind"] == "fully_useable":
                fully_usable_clusters.append(nr)
                occlusion_dicts.append(occlusion_dict)

        return fully_usable_clusters, occlusion_dicts

    def _process_multiple_usable_clusters(
        self,
        diff_img: NDArray[np.float64],
        clusters_in: List[NDArray[np.float64]],
        fully_usable_clusters: List[int],
        occlusion_dicts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process multiple fully usable clusters by selecting the largest one.

        Args:
            diff_img: The difference image.
            clusters_in: List of incoming clusters.
            fully_usable_clusters: Indices of fully usable clusters.
            occlusion_dicts: List of occlusion dictionaries.

        Returns:
            Dict containing information about the detected dart.
        """
        cluster_sizes = []
        for idx, nr in enumerate(fully_usable_clusters):
            cluster_sizes.append(len(clusters_in[nr]))

        biggest_cluster_idx = fully_usable_clusters[np.argmax(cluster_sizes)]
        self.occlusion_dict = occlusion_dicts[biggest_cluster_idx]

        return self.incoming_cluster_detected(
            diff_img, clusters_in[biggest_cluster_idx]
        )

    def _process_overlapping_clusters(
        self, diff_img: NDArray[np.float64], clusters_in: List[NDArray[np.float64]]
    ) -> Optional[Dict[str, Any]]:
        """
        Process clusters that overlap with existing darts.

        Args:
            diff_img: The difference image.
            clusters_in: List of incoming clusters.

        Returns:
            Optional[Dict[str, Any]]: Information about detected dart if successful.
        """
        overlapping_darts, overlap_points = check_overlap(
            np.vstack(clusters_in), self.saved_darts
        )

        if len(overlapping_darts) == 0:
            LOGGER.warning(
                f"Unknown object detected: {len(clusters_in)=}, {len(self.clusters_out)=}"
            )
            return None

        overlap_points = np.vstack(overlap_points)
        which_side_overlap, occluded_rows_clusters = (
            check_which_sides_are_occluded_of_the_clusters(clusters_in, overlap_points)
        )

        angle_of_clusters = calculate_angle_of_different_clusters(
            diff_img,
            clusters_in,
            which_side_overlap,
            occluded_rows_clusters,
        )

        combined_clusters = combine_clusters_based_on_the_angle(
            clusters_in, angle_of_clusters
        )

        return self._handle_combined_clusters(diff_img, clusters_in, combined_clusters)

    def _handle_combined_clusters(
        self,
        diff_img: NDArray[np.float64],
        clusters_in: List[NDArray[np.float64]],
        combined_clusters: List[NDArray[np.float64]],
    ) -> Dict[str, Any]:
        """
        Handle the result of combined clusters analysis.

        Args:
            diff_img: The difference image.
            clusters_in: Original list of incoming clusters.
            combined_clusters: Combined clusters based on angle analysis.

        Returns:
            Dict containing information about the detected dart.
        """
        if len(combined_clusters) == 1:
            # Only middle occluded
            return self.incoming_cluster_detected(diff_img, np.vstack(clusters_in))

        elif len(combined_clusters) == 2:
            distinct_clusters = combined_clusters

        elif len(combined_clusters) == 0 and len(clusters_in) == 2:
            distinct_clusters = [clusters_in[0], clusters_in[1]]

        else:
            LOGGER.warning(
                f"Unexpected overlapping case occurred: {len(clusters_in)=}, {len(combined_clusters)=}"
            )
            return None

        # Choose which cluster is the new dart (selecting the bigger one)
        if len(distinct_clusters) == 2:
            if np.size(distinct_clusters[0]) > np.size(distinct_clusters[1]):
                new_dart_cluster = distinct_clusters[0]
            else:
                new_dart_cluster = distinct_clusters[1]

            return self.incoming_cluster_detected(diff_img, new_dart_cluster)

        return None

    def leaving_cluster_detected(
        self, diff_img: NDArray[np.float64], cluster_out: NDArray[np.float64]
    ) -> None:
        """
        Process a detected leaving dart cluster.

        Identifies which dart has been removed based on the leaving cluster
        and updates the saved darts accordingly.

        Args:
            diff_img: The difference image containing the cluster.
            cluster_out: Coordinates of the detected leaving cluster.
        """
        removed_dart_nr = single_dart_removed(
            diff_img, cluster_out, self.saved_darts, self.dart_removed_tolerance
        )

        if removed_dart_nr is not None:
            self.saved_darts.pop(removed_dart_nr)
            self.dart -= 1

        return None

    def one_incoming_and_leaving_cluster_detected(
        self,
        diff_img: NDArray[np.float64],
        cluster_in: NDArray[np.float64],
        cluster_out: NDArray[np.float64],
    ) -> Optional[Dict[str, Any]]:
        """
        Process the simultaneous detection of an incoming and leaving cluster.

        Determines if this represents a dart movement or a new dart. If it's a movement,
        returns None; otherwise processes it as a new incoming dart.

        Args:
            diff_img: The difference image containing the clusters.
            cluster_in: Coordinates of the detected incoming cluster.
            cluster_out: Coordinates of the detected leaving cluster.

        Returns:
            Optional[Dict[str, Any]]: Information about the detected dart if it's a new dart.
        """
        if dart_moved(
            diff_img, cluster_in, cluster_out, self.dart_moved_difference_thresh
        ):
            return None
        else:
            return self.incoming_cluster_detected(diff_img, cluster_in)

    def get_current_dart_values(self) -> Dict[str, Any]:
        """
        Get the values of the current dart.

        Returns a dictionary with the relevant attributes of the current dart.

        Returns:
            Dict[str, Any]: Dictionary containing dart information:
                - dart: The dart identifier
                - pos: The dart position
                - angle: The dart angle
                - r: The dart radius
                - support: The support value
                - error: The error value
        """
        relevant_outputs = ["dart", "pos", "angle", "r", "support", "error"]

        if self.dart == 0:
            dart_dict = {key: None for key in relevant_outputs}
        else:
            dart_dict = self.saved_darts[f"d{self.dart}"]
            dart_dict["dart"] = f"d{self.dart}"

        return {key: dart_dict.get(key) for key in relevant_outputs}

    def visualize_stream(self, ax: Optional[Axes] = None) -> None:
        """
        Visualize the current image with detected darts.

        Displays the current image and overlays the detected darts with their
        clusters, angles, and positions.

        Args:
            ax: The matplotlib axes to draw on. If None, a new figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax.clear()
        if self.current_img is not None:
            ax.imshow(self.current_img, cmap="gray")

        colors = [
            "aquamarine",
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "brown",
            "orange",
            "pink",
            "olive",
        ]

        try:
            for dart, dart_dict in self.saved_darts.items():
                cluster_in = dart_dict["cluster"]
                cluster_mod = dart_dict["cluster_mod"]
                diff_img = dart_dict["img_pre"]
                angle = dart_dict["angle"]
                pos = dart_dict["pos"]

                # Get color index based on dart number
                color_idx = int(dart[-1]) % len(colors)

                # Plot the original cluster points
                ax.scatter(
                    cluster_in[:, 1],
                    cluster_in[:, 0],
                    c=colors[color_idx],
                    s=2,
                    marker="x",
                    alpha=0.4,
                    label=dart,
                )

                # Plot the modified cluster points
                ax.scatter(
                    cluster_mod[:, 1],
                    cluster_mod[:, 0],
                    c=colors[color_idx],
                    s=10,
                    alpha=0.6,
                    marker="x",
                )

                # Plot the dart angle line
                self._plot_dart_angle_line(ax, diff_img, angle, pos)

            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        except Exception as e:
            LOGGER.warning(f"Error in visualize_stream: {e}")

    def _plot_dart_angle_line(
        self, ax: Axes, diff_img: NDArray[np.float64], angle: float, pos: float
    ) -> None:
        """
        Plot a line representing the dart angle.

        Args:
            ax: The matplotlib axes to draw on.
            diff_img: The difference image used for dimensions.
            angle: The angle of the dart.
            pos: The position of the dart.
        """
        y = np.arange(0, np.shape(diff_img)[0])
        x = (
            np.tan(np.radians(angle * -1)) * y
            + pos
            - np.tan(np.radians(angle * -1)) * max(y)
        )
        ax.plot(
            x,
            y,
            color="cyan",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
        )

    def reset(self) -> None:
        """
        Reset the object to its initial state.

        Clears all stored images, dart information, and counters.
        """
        self.image_count = 0
        self.imgs = []
        self.current_img = None
        self.saved_darts = {}
        self.nr_imgs_distance = 1
        self.dart = 0
        self.occlusion_dict = {}


if __name__ == "__main__":
    from dartdetect.singlecamdartlocalize.simulationtestutils import (
        generate_test_images,
    )

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
