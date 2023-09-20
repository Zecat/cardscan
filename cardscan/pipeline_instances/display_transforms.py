from cv2 import transform
from cardscan.pipeline_instances.rect_perspective_pipeline import *


def draw_contours(contours):
    img = card_contours_transform.initial_input
    if contours is None:
        return img.copy()
    img = copy_bgr(img)
    return [cv2.drawContours(img, contours, -1, (0, 255, 255), 3)]


def draw_contours_def(contours_def):
    c, h = contours_def
    return draw_contours(c)


def debug_image(img, *_):
    return [img]


def debug_images(imgs, *_):
    return imgs


transform_to_img = {
    inner_unique_contours_transform: draw_contours,
    find_contours_def: draw_contours_def,
    approx_filter_quad: draw_contours,
    filter_contours_by_size_solidity: draw_contours,
    card_contours_transform: draw_contours,
    copy_gray: debug_image,
    canny_transform: debug_image,
    rotate_top_left_corner_low_density_transform: debug_images,
    selective_adaptative_threshold: debug_images,
    card_images: debug_images,
    perspective_crop_contours: debug_images,
    close_range_card_contours_detection: draw_contours,
    long_range_card_contours_detection: draw_contours,
    card_contours_detection: draw_contours,
}


def get_debug_imgs(results: List, keep_results: List):
    tiles = []
    for output, transform in zip(results, keep_results):
        imgs = transform_to_img[transform](output)
        # TODO clear err message if transform not found
        for img in imgs:
            tiles.append(img)
    return tiles
