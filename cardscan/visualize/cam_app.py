import sys
import cv2
from cardscan.pipeline import Pipeline
from cardscan.image_processing import *
from typing import Callable

from cardscan import (
    scan,
    card_contours_transform,
    rotate_top_left_corner_low_density_transform,
)
from cardscan.pipeline_instances import transform_to_img


def run_cam_app(fn: Callable):
    cam = cv2.VideoCapture(0)
    window_name = "Track users"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        window_rect = cv2.getWindowImageRect(window_name)
        window_width = window_rect[2]
        window_height = window_rect[3]
        window_size = (window_width, window_height)

        ret, frame = cam.read()
        # frame = cv2.flip(frame, 1)
        if not ret:
            break

        img = fn(frame, window_size)
        cv2.imshow(window_name, img)

        key = cv2.waitKey(1)
        ESC_KEY = 27
        if key == ESC_KEY or key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    sys.exit()
