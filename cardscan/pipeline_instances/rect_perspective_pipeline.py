from cardscan.pipeline import Pipeline, PipelineChoice
import cv2

from cardscan.image_processing import *
from cardscan.image_processing.contour_hierarchy import inner_unique_contours

inner_unique_contours_transform = lambda contours_def: inner_unique_contours(
    contours_def[1], contours_def[0]
)

find_contours_def = lambda img: cv2.findContours(
    img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)


def approx_filter_quad(contours):
    return filter_4_edges_contours(contours_to_poly(contours))


card_contours_transform = Pipeline(
    (
        find_contours_def,
        inner_unique_contours_transform,
        filter_contours_by_size_solidity,
        approx_filter_quad,
    )
)


def selective_adaptative_threshold(img, *_):
    """Adaptative threshold applied on the non-dark area - black otherwise - to avoid holes in the borders if they are too thick."""

    # Define the size of the local neighborhood for calculating the mean
    block_size = 5

    # Calculate the mean of the local neighborhood for each pixel
    mean_values = cv2.boxFilter(img, ddepth=-1, ksize=(block_size, block_size))

    # Define a custom threshold value below which pixels become black
    custom_threshold = 85

    # Create a mask to identify pixels where the mean value is below the custom threshold
    custom_mask = mean_values < custom_threshold

    # Apply adaptive thresholding only to pixels where the custom mask is False
    adaptive_thresholded = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 20
    )

    # TODO compare and benchmark this other way to do it
    # adaptive_thresholded = cv2.adaptiveThreshold(
    #    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 30
    # )

    # Combine the adaptive thresholding result with the custom thresholding result
    result = np.where(custom_mask, 0, adaptive_thresholded)
    return result


def perspective_crop_transform(contours, img):
    return [perspective_crop(contour, img) for contour in contours]


rotate_top_left_corner_low_density_transform = lambda imgs: [
    rotate_top_left_corner_low_density(img, corner_width_ratio=0.1) for img in imgs
]


def canny_transform(img):
    return cv2.Canny(img, threshold1=150, threshold2=255, apertureSize=3)


card_images = Pipeline()


def gaussian_transform(img):
    return cv2.GaussianBlur(img, (5, 5), 1)


close_range_card_contours_detection = (canny_transform, card_contours_transform)
long_range_card_contours_detection = (
    selective_adaptative_threshold,
    card_contours_transform,
)

card_contours_detection = card_images.choice(
    close_range_card_contours_detection,
    long_range_card_contours_detection,
    first_where=len,
)


perspective_crop_contours = card_images.bind_input(perspective_crop_transform)
card_images.setTransforms(
    (
        copy_gray,
        card_contours_detection,
        perspective_crop_contours,
        rotate_top_left_corner_low_density_transform,
    )
)


# The blur option help smoth artefact when detecting from a compressed image
def scan(
    *args,
    blur: bool = False,
    **kwargs,
):
    if blur:
        card_images.transforms = (gaussian_transform,) + card_images.transforms
    return card_images.run(*args, **kwargs)
