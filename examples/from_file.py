from cardscan.pipeline_instances import card_images
from cardscan import (
    scan,
    copy_gray,
    perspective_crop_contours,
    card_contours_detection,
    canny_transform,
    gaussian_transform,
    rotate_top_left_corner_low_density_transform,
)
from cardscan.pipeline_instances.display_transforms import get_debug_imgs
from cardscan.visualize import create_tile_gallery

import cv2
from cv2.typing import MatLike
import sys


def run(filepath: str):
    print(filepath)
    window_name = "Find card"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    img: MatLike = cv2.imread(filepath)
    keep_results = [
        copy_gray,
        card_contours_detection,
        rotate_top_left_corner_low_density_transform,
    ]
    # Add gaussian blur to smooth image compression artefact
    results = scan(img, blur=True, keep_results=keep_results)
    # The previous line is a shortcut for:
    # card_images.transforms = (gaussian_transform,) + card_images.transforms
    # results = card_images.run(img, keep_results=keep_results)
    tiles = get_debug_imgs(results, keep_results)

    (h, w) = img.shape[:2]
    tile_ratio = w / h

    output_img = create_tile_gallery(
        tiles, tile_ratio, (1200, 800)
    )  # TODO dynamic size
    cv2.imshow(window_name, output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        run(filepath)
    else:
        print("No arguments provided.")
