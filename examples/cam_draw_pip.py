from cardscan.visualize import run_pip_app
from cardscan.pipeline_instances import card_images

from cardscan import (
    card_contours_transform,
    rotate_top_left_corner_low_density_transform,
)


def run():
    run_pip_app(
        card_images,
        [
            card_contours_transform,
            rotate_top_left_corner_low_density_transform,
        ],
    )


if __name__ == "__main__":
    run()
