from cardscan.pipeline import Pipeline, Transform
import cv2

from cardscan.image_processing import *
from cardscan.image_processing.contour_hierarchy import inner_unique_contours


def draw_contours(img, contours):
    if contours is None:
        return img.copy()
    img = copy_bgr(img)
    return [cv2.drawContours(img, contours, -1, (0, 255, 255), 3)]


def debug_contours(contours, parent_pipeline):
    base_img = parent_pipeline.initial_input_getter()
    return draw_contours(base_img, contours)


def debug_contours_def(contours_def, parent_pipeline):
    c, h = contours_def
    img = parent_pipeline.initial_input_getter()
    return draw_contours(img, c)


def debug_contours_map_def(contours_map_def, parent_pipeline):
    cmap, h = contours_map_def
    img = parent_pipeline.initial_input_getter()
    c = list(cmap.values())
    return draw_contours(img, c)


def debug_identity(arg, pipeline):
    return arg


inner_unique_contours_transform = Transform(
    lambda contours_def, *_: inner_unique_contours(contours_def[1], contours_def[0]),
    label="Find unique inner contours",
    debug=debug_contours,
)

find_contours_def = Transform(
    lambda img, *_: cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE),
    label="Find contours",
    debug=debug_contours_def,
)

approx_filter_quad = Transform(
    lambda contours, _: filter_4_edges_contours(contours_to_poly(contours)),
    label="Approximate quadrilater contours",
    debug=debug_contours,
)

filter_contours_size_solidity = Transform(
    lambda contours, *_: filter_contours_by_size_solidity(contours),
    label="Filter contour by size and solidity",
    debug=debug_contours,
)

card_contours_transform = Pipeline(
    [
        find_contours_def,
        inner_unique_contours_transform,
        filter_contours_size_solidity,
        approx_filter_quad,
    ],
    label="quadrilater black frame contours pipeline",
)

gray_transform = Transform(
    lambda img, *_: copy_gray(img),
    label="Gray",
    debug=lambda img, *_: [img],
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


reveal_borders_threshold_transform = Transform(
    selective_adaptative_threshold,
    label="Selective adaptative mean threshold",
    debug=lambda img, *_: [img],
)


perspective_crop_transform = Transform(
    lambda contours, pipeline, *_: [
        perspective_crop(contour, pipeline.initial_input_getter())
        for contour in contours
    ],
    label="Perspective crop",
    debug=lambda imgs, *_: imgs,
)

rotate_top_left_corner_low_density_transform = Transform(
    lambda imgs, *_: [
        rotate_top_left_corner_low_density(img, corner_width_ratio=0.1) for img in imgs
    ],
    label="Rotate top left corner low density",
    debug=lambda imgs, *_: imgs,
)

canny_transform = Transform(
    lambda img, *_: cv2.Canny(img, threshold1=150, threshold2=255, apertureSize=3),
    label="Canny",
    debug=lambda arg, *_: [arg],
)

card_images = Pipeline(
    [
        gray_transform,
        reveal_borders_threshold_transform,
        # or canny_transform,
        card_contours_transform,
        perspective_crop_transform,
        rotate_top_left_corner_low_density_transform,
    ],
    label="Find polygone border pipeline",
)

scan = card_images.run
