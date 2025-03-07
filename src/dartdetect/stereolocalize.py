import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import svd
from dartdetect.dartboardgeometry import DartboardGeometry
import dartdetect.calibration.saveandloadcalibdata as sl_calib

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def combine_rt_homogen(R, T):
    """Combines the rotation and translation matrix to a 3x3 matrix in
     homogeneous coordinates.
    Args:
        R: np.array, 2x2, rotation matrix
        T: np.array, 2x1, translation vector
    Returns:
        RT: np.array, 3x3, combined rotation and translation matrix in homogeneous coordinates
    """
    T = T.reshape(2, 1)
    RT = np.hstack((R, T))
    RT = np.vstack((RT, [0, 0, 1]))
    return RT


def projectionmatrics(l_mtx, r_mtx, R_l2d, T_l2d):
    """Calculates the projection matrices of the left and right camera.
    Args:
        l_mtx: np.array, 3x3, intrinsic matrix of the left camera
        r_mtx: np.array, 3x3, intrinsic matrix of the right camera
        R_l2d: np.array, 3x1, rotation matrix from the left camera to the 2D world coordinate system
        T_l2d: np.array, 3x1, translation vector from the left camera to the 2D world coordinate system
    Returns:
        Pl: np.array, 2x3, projection matrix of the left camera
        Pr: np.array, 2x3, projection matrix of the right camera
    """
    l_mtx2d = np.array([[l_mtx[0, 0], l_mtx[0, 2], 0], [0, 1, 0]])
    tr_l = np.eye(3)
    Pl = l_mtx2d @ tr_l
    r_mtx2d = np.array([[r_mtx[0, 0], r_mtx[0, 2], 0], [0, 1, 0]])
    tr_r = np.vstack((np.concatenate([R_l2d, T_l2d], axis=-1), [0, 0, 1]))
    Pr = r_mtx2d @ tr_r
    return Pl, Pr


def DLT(Pl, Pr, ul, ur):
    """Direct Linear Transformation (DLT) algorithm to solve for the 2D point X
    Args:
        Pl: np.array, 3x4, projection matrix of the left camera
        Pr: np.array, 3x4, projection matrix of the right camera
        ul: float, x-coordinate of the point in the left image
        ur: float, x-coordinate of the point in the right image
    Returns:
        x: np.array, 3x1, 2D point in the world coordinate system in homogeneous coordinates
    """
    A = ([Pl[0, :] - ul * Pl[1, :], Pr[0, :] - ur * Pr[1, :]],)
    A = np.array(A).reshape(2, 3)
    B = A.transpose() @ A
    _, _, Vh = svd(B, full_matrices=False)
    x = (Vh[-1, :] / Vh[-1, -1]).reshape(3, 1)
    return x


def C1_to_Cw(tr_c1_cw, c1):
    """Transforms a point from C1 to Cw.
    Args:
        tr_cl_cw: np.array, 3x3, transformation matrix from Cl to Cw
        c1: np.array, 2x1, point in C1
    Returns:
        cr: np.array, 2x1, point in Cw
    """
    cw = tr_c1_cw @ np.vstack((c1.reshape(2, 1), [1]))
    return cw[:2].flatten()


def tr_c1_cw():
    pass


class StereoLocalize(DartboardGeometry):
    def __init__(self, calib_path, fig=None):
        """
        Initializes the DartDetect class.

        Args:
            calib_path(str or path object): path to the calibration folder
                with a dictionary containing camera parameters.
        """
        calib_dict = sl_calib.load_calibration_data(path=calib_path)
        super().__init__(fig)
        self.calib_dict = calib_dict

        R_l2d = self.calib_dict["R_cl_cr_2d"]
        T_l2d = self.calib_dict["T_cl_cr_2d"]

        self.pl, self.pr = projectionmatrics(
            self.calib_dict["l_mtx"], self.calib_dict["r_mtx"], R_l2d, T_l2d
        )
        self.tr_c1_cw = combine_rt_homogen(
            self.calib_dict["R_cl_cw_2d"], self.calib_dict["T_cl_cw_2d"]
        )

    def Cu_to_Cw(self, Cul, Cur):
        """Transforms a point from Cu to Cw.
        Args:
            Cul: float, x-coordinate of the point in the left image
            Cur: float, x-coordinate of the point in the right image
        Returns:
            Cw: np.array, 2x1, point in Cw

        Note:
            self.tr_cl1_cw: np.array, 3x3, transformation matrix from Cl to Cw
            self.pl: np.array, 3x4, projection matrix of the left camera
            self.pr: np.array, 3x4, projection matrix of the right camera
        """

        Cl = DLT(self.pl, self.pr, Cul, Cur)[:2].flatten()
        return -1 * C1_to_Cw(self.tr_c1_cw, Cl)

    def get_dartpoint_from_Cu(self, Cul, Cur):
        """returns the points of an specified location on the dartboard
        Args:
            Cul: float, x-coordinate of the point in the left image
            Cur: float, x-coordinate of the point in the right image
        Returns:
            int: scored point
        """
        xy = self.Cu_to_Cw(Cul, Cur)
        point = self.get_dartpoint_from_cart_coordinates(xy[0], xy[1])
        return point

    def plot_dartposition_from_Cu(self, Cul, Cur, nr="", color="navy"):
        """
        Plot the position of a dart on a graph.

        Args:
            Cul (float): The left camera's Cx coordinate of the dart.
            Cur (float): The right camera's Cx coordinate of the dart.
            nr (str, optional): The dart number/ label to plot. Defaults to "".
            color (str, optional): The color of the plotted dart. Defaults to "navy".

        Returns:
            The plotted dart position.
        """
        xy = self.Cu_to_Cw(Cul, Cur)
        return super().plot_dartposition(xy[0], xy[1], nr, color)


if __name__ == "__main__":
    import pathlib

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    path = "data/calibration_matrices"
    SL = StereoLocalize(path)

    fig = SL.plot_dartboard_emtpy()
    fig.subplots_adjust(bottom=0.25)

    # Create two sliders for adjusting x and y
    axcolor = "lightgoldenrodyellow"
    ax_x = plt.axes([0.3, 0.1, 0.5, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.3, 0.05, 0.5, 0.03], facecolor=axcolor)

    slider_ul = Slider(ax_x, "Cu left cam", 0, 1240.0, valinit=620)
    slider_ur = Slider(ax_y, "Cu right cam", 0, 1240.0, valinit=620)

    # Add a button for updating the plot
    ax_button = plt.axes([0.16, 0.15, 0.3, 0.04])
    button = Button(ax_button, "Throw Dart", color=axcolor, hovercolor="0.975")
    ax_button_clear = plt.axes([0.56, 0.15, 0.3, 0.04])
    button_clear = Button(ax_button_clear, "Clear", color=axcolor, hovercolor="0.975")

    dart_point_text = None
    i = 1

    def update_plot(event):
        global dart_point_text
        global i
        ul = slider_ul.val
        ur = slider_ur.val
        dart_point = SL.get_dartpoint_from_Cu(ul, ur)
        if dart_point_text:
            dart_point_text.remove()
        dart_point_text = plt.text(
            0.5,
            0.95,
            f"Score: {dart_point}",
            transform=plt.gcf().transFigure,
            fontsize=20,
            horizontalalignment="center",
        )

        fig = SL.plot_dartposition_from_Cu(ul, ur, nr=f" D{i} ({ul:.1f}, {ur:.1f})")
        fig.canvas.draw_idle()
        i += 1

    def clear_plot(event):
        global dart_point_text
        global i
        if dart_point_text:
            dart_point_text.remove()
        i = 0
        fig = SL.plot_dartboard_emtpy()
        fig.canvas.draw_idle()
        dart_point_text = None

    button.on_clicked(update_plot)
    button_clear.on_clicked(clear_plot)

    plt.show()
