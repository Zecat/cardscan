import math
from cv2.typing import MatLike
from typing import Tuple
import cv2
import numpy as np

from cardscan.image_processing import copy_bgr


def grid_layout_tile_size(tile_ratio, tile_count, window_size):
    window_w, window_h = window_size
    new_tile_width = new_tile_height = rows = cols = -1
    for cols in range(1, 30):
        new_tile_width = window_w // cols
        new_tile_height = int(new_tile_width / tile_ratio)
        rows = math.ceil(tile_count / cols)
        if rows * new_tile_height <= window_h:
            break
    return (new_tile_width, new_tile_height), (rows, cols)


def scaleWidthKeepRatio(frame: MatLike, bounding_size: Tuple[int, int]) -> MatLike:
    bound_w, bound_h = bounding_size
    frame_h, frame_w = frame.shape[:2]
    scale_factor_fit_w = bound_w / frame_w
    scale_factor_fit_h = bound_h / frame_h
    scale_factor = min(scale_factor_fit_w, scale_factor_fit_h)
    resized_frame = cv2.resize(
        frame,
        (int(frame_w * scale_factor), int(frame_h * scale_factor)),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized_frame


def get_top_left(i: int, cols: int, size: Tuple[int, int]):
    col_i = i % cols
    row_i = i // cols
    return col_i * size[0], row_i * size[1]


def create_tile_gallery(
    tiles, img_ratio: float, window_size: Tuple[int, int]
) -> MatLike:
    tile_bounding_size, (tot_rows, tot_cols) = grid_layout_tile_size(
        img_ratio, len(tiles), window_size
    )

    tiles = [scaleWidthKeepRatio(tile, tile_bounding_size) for tile in tiles]
    tiles_top_left = [
        get_top_left(i, tot_cols, tile_bounding_size) for i, tile in enumerate(tiles)
    ]

    window_w, window_h = window_size
    img_shape = (window_h, window_w, 3)
    img = np.zeros(img_shape, dtype=np.uint8)

    for tile, (x, y) in zip(tiles, tiles_top_left):
        h, w = tile.shape[:2]
        img[y : y + h, x : x + w] = copy_bgr(tile)
    return img
