import cv2
from cv2.typing import MatLike


def has_3_channels(img: MatLike) -> bool:
    return img.shape[-1] == 3


def copy_rgb(img: MatLike) -> MatLike:
    """Convert from BGR or GRAY ro RGB"""
    if has_3_channels(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def copy_bgr(img: MatLike) -> MatLike:
    """If image is 3 channel, just copy. Else, convert from GRAY to BGR"""
    if has_3_channels(img):
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def copy_gray(img: MatLike) -> MatLike:
    """If image is 3 channel, convert from GRAU to BGR. Else, just copy."""
    if has_3_channels(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()
