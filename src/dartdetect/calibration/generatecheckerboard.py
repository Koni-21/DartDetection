import logging
import cv2
import numpy as np


def find_square_size(pixels_h, pixels_v, diag):
    """
    Finds the square size in centimeters and possible dimensions to fit
    a checkerboard pattern exactly in the given screen.

    Args:
        pixels_h (int): Number of horizontal pixels.
        pixels_v (int): Number of vertical pixels.
        diag (float): Diagonal length of the checkerboard pattern in centimeters.

    Returns:
        tuple: A tuple containing a dictionary and a tuple. The first dictionary
            contains the possible dimensions of the checkerboard pattern,
            where the keys are indices and the values are tuples of
            (square_size_cm, square_size_px, rows, columns).
            The second tuple contains the calculated vertical
            and horizontal lengths in centimeters.
    """
    aspect_ratio = pixels_h / pixels_v
    posible_square_sizes = []
    for i in range(1, 1000):
        if pixels_h % i == 0:
            if pixels_v % i == 0:
                if pixels_h / i == pixels_v / i * aspect_ratio:
                    posible_square_sizes.append(i)

    v_cm = diag / (np.sqrt(aspect_ratio**2 + 1))
    h_cm = v_cm * aspect_ratio

    square_size_cm = {
        posible_square_size: posible_square_size * v_cm / pixels_v
        for posible_square_size in posible_square_sizes
    }
    # only look at square sizes between 0.5 cm and 10 cm
    usefull_square_sizes = {
        square_size: square_size_cm[square_size]
        for square_size in square_size_cm
        if square_size_cm[square_size] > 0.5 and square_size_cm[square_size] < 10
    }

    possible_dimesions = {}
    for i, (square_size_px, square_size_cm) in enumerate(usefull_square_sizes.items()):
        rows = pixels_v / square_size_px
        columns = pixels_h / square_size_px
        possible_dimesions[i] = (square_size_cm, square_size_px, rows, columns)

    return possible_dimesions, (v_cm, h_cm)


def generate_chessboard(pixels_h, rows, columns):
    """
    Generate a chessboard image with the specified number of rows and columns.

    Args:
        pixels_h (int): The horizontal number of pixels of the chessboard image.
        rows (int): The number of rows in the chessboard.
        columns (int): The number of columns in the chessboard.

    Returns:
        numpy.ndarray: The generated chessboard image.

    Raises:
        ValueError: If the number of columns or rows is not an integer value,
                    or if the horizontal pixels are not evenly divisible by the columns.
    """
    if columns % 1 == 0:
        columns = int(columns)
    else:
        raise ValueError(
            "Column number {columns} is not a integer value: columns % 1 != 0"
        )
    if rows % 1 == 0:
        rows = int(rows)
    else:
        raise ValueError("Row number {rows} is not a integer value: columns % 1 != 0")
    # Define the size of each square in pixels
    if pixels_h % columns == 0:
        square_size = int(pixels_h / columns)
    else:
        raise ValueError(
            f"Horizontal pixels {pixels_h} are not evenly"
            f" devisable with the columns {columns}"
        )

    # Create an empty chessboard image
    chessboard = np.zeros(
        (rows * square_size, columns * square_size, 3), dtype=np.uint8
    )
    for row in range(rows):
        for col in range(columns):
            # Determine the color of the square based on row and column indices
            if (row + col) % 2 == 0:
                square_color = (255, 255, 255)  # white
            else:
                square_color = (0, 0, 0)  # black

            # Calculate the coordinates of the square
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size

            # Draw the square on the chessboard image
            cv2.rectangle(chessboard, (x1, y1), (x2, y2), square_color, -1)

    return chessboard


def remove_upper_row(chessboard, square_size, rows=1):
    chessboard[int(rows - 1) : square_size, :, :] = (0, 0, 0)


def remove_lower_row(chessboard, square_size, rows=1):
    chessboard[-int(rows * square_size) :, :, :] = (0, 0, 0)


def save_chessboard(name, chessboard, rows, columns, square_size):
    cv2.imwrite(
        f"{name}_chessboard_r{rows-3}c{columns}s{int(square_size*1000)}mm.jpg",
        chessboard,
    )


if __name__ == "__main__":
    # sizes Surface
    pixels_h = 2736
    pixels_v = 1824
    diag = 31.242  # cm
    posible_square_sizes, (v_cm, h_cm) = find_square_size(pixels_h, pixels_v, diag)
    print("Possible square sizes Surface: \n")
    [
        print(f"{k}: Square: {v[0]:.4f} cm/ {v[1]} px, rows {v[2]}, columns {v[3]}")
        for k, v in posible_square_sizes.items()
    ]
    print("Size surface v,h [cm]", v_cm, h_cm)

    # sizes Pixel 4
    pixels_h = 1080
    pixels_v = 2340
    diag = 14.732  # cm

    posible_square_sizes, (v_cm, h_cm) = find_square_size(pixels_h, pixels_v, diag)

    print("\nPossible square sizes Pixel4a: \n")
    [
        print(f"{k}: Square: {v[0]:.4f} cm/ {v[1]} px, rows {v[2]}, columns {v[3]}")
        for k, v in posible_square_sizes.items()
    ]
    print("Size pixel v,h [cm]", v_cm, h_cm)

    choose_size = 1
    rows = posible_square_sizes[choose_size][2]
    columns = posible_square_sizes[choose_size][3]
    square_size_cm = posible_square_sizes[choose_size][0]
    square_size_px = posible_square_sizes[choose_size][1]

    chessboard = generate_chessboard(pixels_h, rows, columns)
    print(f"rows, colums: {rows-3}, {columns}, size: {int(square_size_cm*10000)} um")

    remove_upper_row(chessboard, square_size_px, rows=1)
    remove_lower_row(chessboard, square_size_px, rows=2)

    cv2.imshow("Chessboard", chessboard)
    cv2.waitKey(0)
