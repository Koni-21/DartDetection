import numpy as np


## simulation functions
def draw_dart_subpixel(img, x_pos, angle, width):
    """
    Generate an image with a black bar of adjustable position, angle, and thickness.
    The image is represented as a NumPy array with subpixel accuracy achieved via
    grayscale gradients.

    Args:
        img (np.array): input image.
        x_pos (float): Horizontal position of the bar's center (subpixel accuracy).
        angle (float): Angle of the bar in degrees (clockwise).
        width (int): Thickness of the bar in pixels.
        array_shape (tuple): Shape of the array (height, width).

    Returns:
        np.ndarray: The generated image as a 2D array.
    """

    height, width_img = img.shape
    # Convert angle to radians
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Loop through each pixel and compute its distance to the bar
    for y in range(height):
        for x in range(width_img):
            # Rotate pixel coordinates
            x_rot = (x - x_pos) * cos_angle + (y - height + 1) * sin_angle

            distance = abs(x_rot)  # Perpendicular distance to the center of the bar

            # Apply intensity gradient based on distance
            if distance <= width / 2:
                img[y, x] = 0  # Fully black inside the bar
            elif distance <= width / 2 + 1:  # Transition zone for subpixel accuracy
                gradient_val = distance - width / 2  # Linear gradient [0, 1]
                img[y, x] = (
                    np.clip(gradient_val + img[y, x], 1, 2) - 1
                )  # if already gray value: sum up

    return img


def generate_test_images(
    img=np.ones([5, 20]),
    positions=[5, 10, 15],
    angles=[3, 2, 9],
    widths=[2, 2, 3],
    move_darts=[0, 0, 1],
):
    imgs = []

    imgs.append(img.copy())

    for i in range(0, len(positions)):
        imgs.append(img.copy())

        if i >= 1:
            if move_darts[i] == 1:
                img[:, :] = 1
                if i > 1:
                    if move_darts[i - 1] == 0:
                        for j in range(0, i - 1):
                            img = draw_dart_subpixel(
                                img.copy(), positions[j], angles[j], widths[j]
                            )

        img = draw_dart_subpixel(img.copy(), positions[i], angles[i], widths[i])
        imgs.append(img.copy())
        imgs.append(img.copy())
    return imgs


def add_noise(img, noise):
    img -= np.random.random_sample(np.shape(img)) * noise
    return np.clip(img, 0, 1)
