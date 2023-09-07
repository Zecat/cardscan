#!/usr/bin/env python3

import cv2
from cv2.typing import MatLike
import numpy as np

from cardscan import scan, card_contours, card_images

from cardscan.image_processing import copy_bgr


def scaleWidthKeepRatio(frame: MatLike, desired_width: int) -> MatLike:
    frame_height, frame_width, _ = frame.shape
    scale_factor = desired_width / frame_width
    resized_frame = cv2.resize(
        frame,
        (desired_width, int(frame_height * scale_factor)),
    )
    return resized_frame


def draw_contours(img, contours):
    if contours is None:
        return img.copy()
    img = copy_bgr(img)
    return cv2.drawContours(img, contours, -1, (0, 255, 255), 3)


def run():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("cardscan", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("cardscan", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Adjust the size as needed

    while True:
        ret, cam_frame = cam.read()
        if not ret:
            break

        contours, final_imgs = scan(
            cam_frame,
            results=[card_contours, card_images],
        )
        if len(final_imgs):
            frame = final_imgs[0]

        cam_frame = draw_contours(cam_frame, contours)

        cam_frame = scaleWidthKeepRatio(cam_frame, 200)
        h, w = cam_frame.shape[:2]

        # Overlay the smaller image onto the larger one
        frame[0:h, 0:w] = cam_frame

        # Create an empty frame with the same dimensions
        cv2.imshow("cardscan", frame)

        key = cv2.waitKey(1)
        ESC_KEY = 27
        if key == ESC_KEY or key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    return None


if __name__ == "__main__":
    run()
