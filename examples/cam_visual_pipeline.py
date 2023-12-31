from cardscan.visualize import run_pip_app
from cardscan.pipeline_instances import card_images

from cardscan import *


def run():
    run_pip_app(
        card_images,
        [
            copy_gray,
            find_contours_def,
            inner_unique_contours_transform,
            filter_contours_by_size_solidity,
            approx_filter_quad,
            card_contours_transform,
            perspective_crop_contours,
            rotate_top_left_corner_low_density_transform,
        ],
    )


if __name__ == "__main__":
    run()
