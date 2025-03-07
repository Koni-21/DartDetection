import logging
import itertools

import numpy as np
import matplotlib.pyplot as plt

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
        self.image_empty = None
        self.image_count = 0
        self.imgs = []
        self.current_img = None
        self.current_diff_img = None

        self.saved_darts = {}
        self.nr_imgs_distance = 1
        self.dart = 0
        self.occlusion_dict = {}

        # parameters of the subfunctions
        self.distance_to_bottom_not_arrived = 1  # for dart not yet arrived
        self.dart_removed_tolerance = 1
        self.thresh_n_pixels_dart_cluster = 2  # nr of pixels needed to be a cluster
        self.thresh_binarise_cluster = 0.3
        self.dbscan_eps_cluster = 1
        self.dbscan_min_samples_cluster = 2
        self.thresh_noise = 0.1
        self.dart_moved_difference_thresh = 1

        self.dilate_cluster_by_n_px = 1

        self.min_usable_columns_middle_overlap = 1

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
        if self.image_count == 1:
            self.image_empty = img
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

        diff_img = compare_imgs(imgs[-(self.nr_imgs_distance + 1)], self.current_img)
        self.current_diff_img = diff_img
        clusters_in, clusters_out = try_get_clusters_in_out(
            diff_img,
            thresh_binarise=self.thresh_binarise_cluster,
            thresh_n_pixels_dart=self.thresh_n_pixels_dart_cluster,
            dbscan_eps=self.dbscan_eps_cluster,
            dbscan_min_samples=self.dbscan_min_samples_cluster,
        )
        self.clusters_in, self.clusters_out = clusters_in, clusters_out

        if len(clusters_in) == 0 and len(clusters_out) == 0:
            return None
        elif len(clusters_in) == 1 and len(clusters_out) == 0:
            return self.incoming_cluster_detected(diff_img, clusters_in[0])
        elif len(clusters_in) == 1 and len(clusters_out) == 0:
            return self.one_incoming_and_leaving_cluster_detected(
                diff_img, clusters_in[0], clusters_out[0]
            )
        elif len(clusters_in) > 1:
            return self.multiple_clusters_detected(diff_img, clusters_in, clusters_out)
        elif len(clusters_out) >= 1:
            self.check_view_empty()
            return self.leaving_cluster_detected(diff_img, np.vstack(clusters_out))

    def check_view_empty(self):
        """
        Detects if the view is empty and updates the saved darts accordingly.

        Args:
            diff_img (numpy.ndarray): The difference image used to detect changes.
        Returns:
            None
        """
        diff_img = compare_imgs(self.image_empty, self.current_img)
        clusters_in, clusters_out = try_get_clusters_in_out(
            diff_img,
            thresh_binarise=self.thresh_binarise_cluster,
            thresh_n_pixels_dart=self.thresh_n_pixels_dart_cluster,
            dbscan_eps=self.dbscan_eps_cluster,
            dbscan_min_samples=self.dbscan_min_samples_cluster,
        )
        if len(clusters_out) > 0:
            LOGGER.warning("View of the camera was not emtpy at the start. Restarting.")
            self.reset()
        elif len(clusters_in) == 0:
            LOGGER.info("View of the camera is empty. Restarting.")
            self.reset()
        return None

    def incoming_cluster_detected(self, diff_img, cluster_in):
        """
        Detects an incoming cluster in the given difference image and processes it.
        This method checks if a dart has fully arrived or if two consecutive
        images still show the cluster (Dart does not move anymore).
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
            not dart_fully_arrived(
                np.shape(diff_img)[0], cluster_in, self.distance_to_bottom_not_arrived
            )
            and self.nr_imgs_distance < 2
        ):
            self.nr_imgs_distance = 2
            return None
        self.nr_imgs_distance = 1

        cluster_mod = cluster_in

        cluster_in = dilate_cluster(
            cluster_in, np.shape(diff_img)[1], self.dilate_cluster_by_n_px
        )
        if not self.occlusion_dict:
            self.occlusion_dict = check_occlusion_type_of_a_single_cluster(
                cluster_in, self.saved_darts
            )

        if self.occlusion_dict:
            pos, angle, support, r, error, cluster_mod = (
                calculate_position_from_occluded_dart(
                    self.occlusion_dict,
                    cluster_in,
                    diff_img,
                    self.current_img,
                    self.saved_darts,
                    self.min_usable_columns_middle_overlap,
                )
            )
        else:
            pos, angle, support, r, error = calculate_position_from_cluster_and_image(
                diff_img, cluster_in
            )

        self.dart += 1

        self.saved_darts[f"d{self.dart}"] = {
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
        return self.saved_darts[f"d{self.dart}"]

    def multiple_clusters_detected(self, diff_img, clusters_in, clusters_out):
        LOGGER.warning(
            f"More than one new cluster found: {len(clusters_in)=}, {len(clusters_out)=}). "
        )

        if self.nr_imgs_distance < 2:
            self.nr_imgs_distance = 2
            return None

        # check fully usable clusters:
        fully_usable_clusters = []
        occlusion_dicts = []
        for nr, cluster_in in enumerate(clusters_in):
            cluster_in = dilate_cluster(
                cluster_in, np.shape(diff_img)[1], self.dilate_cluster_by_n_px
            )
            occlusion_dict = check_occlusion_type_of_a_single_cluster(
                cluster_in, self.saved_darts
            )
            if occlusion_dict["occlusion_kind"] == "fully_useable":
                fully_usable_clusters.append(nr)
                occlusion_dicts.append(occlusion_dict)

        if len(fully_usable_clusters) == 1:
            self.occlusion_dict = occlusion_dicts[fully_usable_clusters[0]]
            return self.incoming_cluster_detected(
                diff_img, clusters_in[fully_usable_clusters[0]]
            )
        elif len(fully_usable_clusters) > 1:
            cluster_sizes = []
            for idx, nr in enumerate(fully_usable_clusters):
                cluster_sizes.append(len(clusters_in[fully_usable_clusters[idx]]))

            biggest_cluster_idx = fully_usable_clusters[np.argmax(cluster_sizes)]
            self.occlusion_dict = occlusion_dicts[biggest_cluster_idx]
            return self.incoming_cluster_detected(
                diff_img, clusters_in[biggest_cluster_idx]
            )

        overlapping_darts, overlap_points = check_overlap(
            np.vstack(clusters_in), self.saved_darts
        )
        if len(overlapping_darts) == 0:
            LOGGER.warning(
                f"Unknown object detected: {len(clusters_in)=}, {len(clusters_out)=}"
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

        if len(combined_clusters) == 1:
            # Only middle occluded
            return self.incoming_cluster_detected(diff_img, np.vstack(clusters_in))
        elif len(combined_clusters) == 2:
            distinct_clusters = combined_clusters
        elif len(combined_clusters) == 0:
            if len(clusters_in) == 2:
                distinct_clusters = [clusters_in[0], clusters_in[1]]
        else:
            print(
                f"unexpected overlapping case occurred  {len(clusters_in)=}, {len(combined_clusters)=}"
            )

        if len(distinct_clusters) == 2:
            # choose which cluster is the new: more or less randomly
            # choose the bigger cluster as the new one
            if np.size(distinct_clusters[0]) > np.size(distinct_clusters[1]):
                new_dart_cluster = distinct_clusters[0]
                old_dart_cluster = distinct_clusters[1]
            else:
                new_dart_cluster = distinct_clusters[1]
                old_dart_cluster = distinct_clusters[0]

        return self.incoming_cluster_detected(diff_img, new_dart_cluster)

    def leaving_cluster_detected(self, diff_img, cluster_out):
        """
        Detects if a dart has left the cluster and updates the saved darts accordingly.

        Args:
            diff_img (numpy.ndarray): The difference image used to detect changes.
            cluster_out (numpy.ndarray): The coordinates of the detected cluster.
        Returns:
            None
        """

        removed_dart_nr = single_dart_removed(
            diff_img, cluster_out, self.saved_darts, self.dart_removed_tolerance
        )
        if removed_dart_nr is not None:
            self.saved_darts.pop(removed_dart_nr)
            self.dart -= 1

        return None

    def one_incoming_and_leaving_cluster_detected(
        self, diff_img, cluster_in, cluster_out
    ):
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

    def get_current_dart_values(self):
        """
        Retrieves the current dart values from the saved darts dictionary.

        This method returns a dictionary containing only the relevant outputs
        for the current dart, which include position, angle, radius, support,
        and error.

        Returns:
            dict: A dictionary with keys "dart", "pos", "angle", "r", "support", and "error",
                  containing the corresponding values for the current dart.
        """
        relevant_outputs = ["dart", "pos", "angle", "r", "support", "error"]
        if self.dart == 0:
            dart_dict = {key: None for key in relevant_outputs}
        else:
            dart_dict = self.saved_darts[f"d{self.dart}"]
            dart_dict["dart"] = f"d{self.dart}"

        return {key: dart_dict[key] for key in relevant_outputs}

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

                ax.scatter(
                    cluster_in[:, 1],
                    cluster_in[:, 0],
                    c=colors[int(dart[-1])],
                    s=2,
                    marker="x",
                    alpha=0.4,
                    label=dart,
                )

                ax.scatter(
                    cluster_mod[:, 1],
                    cluster_mod[:, 0],
                    c=colors[int(dart[-1])],
                    s=10,
                    alpha=0.6,
                    marker="x",
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
                    color="cyan",
                    linestyle="-",
                    linewidth=1,
                    alpha=0.8,
                )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        except Exception as e:
            print(f"Skipped error in visualize_stream: {e}")

    def reset(self):
        """
        Resets the SingleCamLocalize object to its initial state.

        This method resets the image count, images list, current image, saved darts,
        distance, and dart count to their initial values.
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
