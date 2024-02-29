import numpy as np
from matplotlib import pyplot as plt


class dartboard_geometry:
    def __init__(self):
        """generate a parametrized dartboard in cm coordinates

        contains a function to read out points of position:
            'get_dartpoint_from_cart_coordinates'
        and functions to plot a the dartboard and the dart location on it
        """
        # Radien in cm angeben, aktuell: Offizielle Werte DDV
        self.radii = dict(
            [
                ["r_double1", 17.08 - 0.1],
                ["r_double2", 16.12 - 0.07],
                ["r_triple1", 10.78 - 0.08],
                ["r_triple2", 9.82],
                ["doublebull", 1.27 / 2],
                ["bull", 3.18 / 2],
            ]
        )

        # Punktzahlen der Reihe nach
        # self.werte = [10,15,2,17,3,19,7,16,8,11,14,9,12,5,20,1,18,4,13,6]
        self.werte = [
            13,
            4,
            18,
            1,
            20,
            5,
            12,
            9,
            14,
            11,
            8,
            16,
            7,
            19,
            3,
            17,
            2,
            15,
            10,
            6,
        ]

        self.fig, self.ax = plt.subplots(figsize=(10, 10))

    def _cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        phi_deg = phi / np.pi * 180
        return (rho, phi_deg)

    def _pol2cart(self, rho, phi_deg):
        phi = phi_deg / 180 * np.pi
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def get_dartpoint_from_cart_coordinates(self, x, y):
        """returns the points of an specified location on the dartboard

        Args:
            x (float): x coordinate of the dart in cm
            y (float): y coordinate of the dart in cm

        Returns:
            int: scored point
        """
        radii = self.radii
        werte = self.werte

        field_angle = int(360 / 20)
        rho, angle = self._cart2pol(x, y)
        point_raw = werte[int(np.round(angle / field_angle)) - 1]

        if rho > radii["r_double1"]:
            point = 0
        elif rho < radii["r_double1"] and rho >= radii["r_double2"]:
            point = point_raw * 2
        elif rho < radii["r_triple1"] and rho >= radii["r_triple2"]:
            point = point_raw * 3
        elif rho < radii["bull"] and rho >= radii["doublebull"]:
            point = 25
        elif rho <= radii["doublebull"]:
            point = 50
        else:
            point = point_raw

        return point

    def plot_dartboard_emtpy(self):
        """function to plot an empty dartboard"""
        fig, ax = self.fig, self.ax
        ax.clear()

        ############ Style Plot ##############

        # Select length of axes and the space between tick labels
        xmin, xmax, ymin, ymax = -20, 20, -20, 20
        ticks_frequency = 5

        # Set identical scales for both axes
        ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect="equal")

        # Set bottom and left spines as x and y axes of coordinate system
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Create 'x' and 'y' labels placed at the end of the axes
        ax.set_xlabel("x [cm]", size=14, x=0.96)  # , labelpad=-35, x=1.03,)
        ax.set_ylabel("y [cm]", size=14, labelpad=-21, y=1.02, rotation=0)

        # Create custom major ticks to determine position of tick labels
        x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
        y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
        ax.set_xticks(x_ticks[x_ticks != 0], minor=False)
        ax.set_yticks(y_ticks[y_ticks != 0], minor=False)

        # Create minor ticks placed at each integer to enable drawing of minor grid
        # lines: note that this has no effect in this example with ticks_frequency=1
        ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
        ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)
        # Draw major and minor grid lines
        ax.grid(which="minor", color="grey", linewidth=1, linestyle="-", alpha=0.2)
        ax.grid(which="major", color="grey", linewidth=1.5, linestyle="-", alpha=0.2)

        # Draw arrows
        arrow_fmt = dict(markersize=4, color="black", clip_on=False)
        ax.plot((1), (0), marker=">", transform=ax.get_yaxis_transform(), **arrow_fmt)
        ax.plot((0), (1), marker="^", transform=ax.get_xaxis_transform(), **arrow_fmt)

        ############### Draw Dartboard ############

        toggle = False
        for field_angle in range(-9, 351, 18):
            field_angle2 = field_angle + 18
            x1, y1 = self._pol2cart(self.radii["r_double1"] + 2, field_angle)
            x2, y2 = self._pol2cart(self.radii["r_double1"] + 2, field_angle2)

            res = 10
            x1 = np.linspace(0, x1, res)
            y1 = np.linspace(0, y1, res)
            x2 = np.linspace(0, x2, res)
            y2 = np.linspace(0, y2, res)

            xf = np.concatenate((x1, x2[::-1]))
            yf = np.concatenate((y1, y2[::-1]))
            if toggle:
                ax.fill(xf, yf, color="dimgray")
                toggle = False
            else:
                ax.fill(xf, yf, color="khaki")
                toggle = True

        # add point text
        for field_angle in range(0, 360, 18):
            x, y = self._pol2cart(self.radii["r_double1"] + 1, field_angle)
            ax.text(
                x,
                y,
                str(self.werte[int(np.round(field_angle / 18)) - 1]),
                fontsize=20,
                color="fuchsia",
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Draw circles
        def get_circle_band(lower_r, upper_r):
            x1 = np.linspace(-lower_r, lower_r, 500)
            x2 = np.linspace(-upper_r, upper_r, 500)
            y1 = np.sqrt(lower_r**2 - x1**2)
            y2 = np.sqrt(upper_r**2 - x2**2)
            # get also lower half of the circle
            x1 = np.concatenate([x1, x1[::-1]])
            x2 = np.concatenate([x2, x2[::-1]])
            y1 = np.concatenate([y1, -y1[::-1]])
            y2 = np.concatenate([y2, -y2[::-1]])
            # stitch both circles to fill
            xf = np.concatenate((x1, x2[::-1]))
            yf = np.concatenate((y1, y2[::-1]))
            return xf, yf

        # cut off edge
        xf, yf = get_circle_band(self.radii["r_double1"], 20)
        ax.fill(xf, yf, color="w")

        xf, yf = get_circle_band(self.radii["r_double2"], self.radii["r_double1"])
        ax.fill(xf, yf, color="g", alpha=0.5)

        xf, yf = get_circle_band(self.radii["r_triple2"], self.radii["r_triple1"])
        ax.fill(xf, yf, color="g", alpha=0.5)

        xf, yf = get_circle_band(self.radii["bull"], self.radii["doublebull"])
        ax.fill(xf, yf, color="olivedrab")
        xf, yf = get_circle_band(self.radii["doublebull"], 0)
        ax.fill(xf, yf, color="tomato")

        return fig

    def plot_dartposition(self, x, y, nr="", color="navy"):
        """marks a given position on the dartboard

        Args:
            x (float): x coordinate of the dart in cm
            y (float): y coordinate of the dart in cm
            nr (str, optional): optional text to show at the dart position. E.g.
                Dart number oder the point scored. Defaults to "".
        """

        fig, ax = self.fig, self.ax
        # ax.scatter(x,y, marker = "x", color = "navy")
        size = 0.2
        ax.plot([x - size, x + size], [y - size, y + size], color=color)
        ax.plot([x - size, x + size], [y + size, y - size], color=color)
        ax.text(x, y, nr, fontsize=12, color=color)

        return fig

    def clear_plot(self):
        self.ax.cla()
        self.plot_dartboard_emtpy()


if __name__ == "__main__":
    x, y = 9, 10

    dart_geo = dartboard_geometry()
    fig = dart_geo.plot_dartboard_emtpy()

    def onkey(event):
        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        print("Dart hit coordinates: x: {:.2f}, y: {:.2f}".format(x, y))
        points = dart_geo.get_dartpoint_from_cart_coordinates(x, y)
        fig = dart_geo.plot_dartposition(x, y, points)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def onbutton(event):
        print("Reload")
        dart_geo.clear_plot()

    cid2 = dart_geo.fig.canvas.mpl_connect("resize_event", onbutton)
    cid1 = dart_geo.fig.canvas.mpl_connect("key_press_event", onkey)

    plt.show()
