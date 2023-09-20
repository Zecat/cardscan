from cv2.typing import MatLike
from typing import Tuple, List

from cardscan.pipeline import Pipeline
from cardscan.image_processing import *

from cardscan.pipeline_instances.display_transforms import get_debug_imgs

from .tile_gallery import create_tile_gallery
from .cam_app import run_cam_app


def run_pip_app(pipeline: Pipeline, keep_results: List):
    def get_screen_img(frame: MatLike, window_size: Tuple[int, int]):
        results = pipeline.run(frame, keep_results=keep_results)
        tiles = get_debug_imgs(results, keep_results)

        (h, w) = frame.shape[:2]
        tile_ratio = w / h
        img = create_tile_gallery(tiles, tile_ratio, window_size)
        return img

    run_cam_app(get_screen_img)
