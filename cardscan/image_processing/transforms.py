import numpy as np
from cv2.typing import MatLike
import cv2
from typing import List, Tuple

Contour = List[Tuple[int, int]]


def perspective_crops(contours: List[Contour], img: MatLike):
    return [perspective_crop(contour, img) for contour in contours]


def perspective_crop(contour: Contour, img: MatLike):
    src_points = contour

    w, h = img.shape[:2]
    top_left_coord = [0, 0]
    top_right_coord = [w, 0]
    bottom_left_coord = [0, w]
    bottom_right_coord = [w, w]

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(
        [top_left_coord, top_right_coord, bottom_right_coord, bottom_left_coord],
        dtype=np.float32,
    )
    transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    w, h = img.shape[:2]
    transformed_image = cv2.warpPerspective(img, transformation_matrix, (w, h))
    cropped_img = transformed_image[0:w, 0:h]
    return cropped_img


def rotate_top_left_corner_low_density(img: MatLike, corner_size: int = 20):
    """Compare the mean intensity of the 4 corners of `img` where each corner is a square with size `corner_size`. Rotate the image so the corner with lowest intensity (more black) is located at the top left."""

    corners_topleft = [
        (0, 0),  # Top-left corner
        (img.shape[1] - corner_size, 0),  # Top-right corner
        (0, img.shape[0] - corner_size),  # Bottom-left corner
        (img.shape[0] - corner_size, img.shape[1] - corner_size),  # Bottom-right corner
    ]
    """The top-left position of corners in order TL-TR-BR-BL"""

    squares = []
    """Corners subimage of the image in respectve order to `corners_topleft`."""

    mean_intensities = []
    """Mean intensity of each corner in respectve order to `corners_topleft`"""

    for corner_x, corner_y in corners_topleft:
        square = img[
            corner_x : corner_x + corner_size, corner_y : corner_y + corner_size
        ]
        mean_intensity = np.mean(square)
        squares.append(square)
        mean_intensities.append(mean_intensity)

    min_intensity_index = np.argmin(mean_intensities)
    """The index of the corner with lowest mean intensity"""

    angles = [0, -90, 90, 180]
    """The rotation angle needed to rotate the image so the index associated corner is located at the top-left of the image. In respectve order to `corners_topleft`."""

    angle = angles[min_intensity_index]
    """ The rotation angle to align the corner with the lowest mean intensity to the top-left"""

    if angle == 0:
        rotated_img = img
    else:
        rows, cols, *_ = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_img = cv2.warpAffine(img, M, (cols, rows))

    return rotated_img
