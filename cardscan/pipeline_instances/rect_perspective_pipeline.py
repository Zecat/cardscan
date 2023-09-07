from cardscan.pipeline import Pipeline, Pipecell
import functools
import cv2

from cardscan.image_processing import *


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


find_quadrilater_black_frame_contours_pipeline = Pipeline(
    [
        Pipecell(
            lambda img, _: cv2.findContours(
                img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            ),
            label="Find contours",
            debug=debug_contours_def,
        ),
        Pipecell(
            lambda contours_def, _: (
                filter_contours_by_size(contours_def[0]),
                contours_def[1],
            ),
            label="Filter contour by size",
            debug=debug_contours_map_def,
        ),
        Pipecell(
            lambda contours_def, _: filter_containing_contours(
                contours_def[0], contours_def[1]
            ),
            label="Filter containing contours",
            debug=debug_contours,
        ),
        Pipecell(
            lambda contours, _: filter_4_edges_contours(contours_to_poly(contours)),
            label="Approximate quadrilater",
            debug=debug_contours,
        ),
    ],
    label="Find quadrilater black frame contours pipeline",
)

canny = functools.partial(cv2.Canny, threshold1=150, threshold2=255, apertureSize=3)

find_polygone_black_frame_pipeline = Pipeline(
    [
        Pipecell(
            lambda img, _: copy_gray(img),
            label="Gray",
            debug=lambda img, pipeline: [img],
        ),
        Pipecell(
            lambda img, _: canny(img),
            label="Canny",
            debug=lambda arg, pipeline: [arg],
        ),
        find_quadrilater_black_frame_contours_pipeline,
        Pipecell(
            lambda contours, pipeline: [
                perspective_crop(contour, pipeline.initial_input_getter())
                for contour in contours
            ],
            label="Perspective crop",
            debug=lambda imgs, _: imgs,
        ),
        Pipecell(
            lambda imgs, _: [
                rotate_top_left_corner_low_density(img, corner_size=50) for img in imgs
            ],
            label="Rotate top left corner low density",
            debug=lambda imgs, pipeline: imgs,
        ),
    ],
    label="Find polygone border pipeline",
)

scan = find_polygone_black_frame_pipeline.run
