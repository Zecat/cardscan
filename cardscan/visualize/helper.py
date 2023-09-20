import cv2


def getCv2WindowSize(window_name: str):
    window_rect = cv2.getWindowImageRect(window_name)
    window_width = window_rect[2]
    window_height = window_rect[3]
    window_size = (window_width, window_height)
