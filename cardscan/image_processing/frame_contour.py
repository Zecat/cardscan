import cv2
from cv2.typing import MatLike


def contour_to_poly(contour):
    """Approximate the contour to a polygon."""
    epsilon = 0.04 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
    return approx_polygon


def contours_to_poly(contours):
    """Approximate the contours to polygones"""
    return [contour_to_poly(contour) for contour in contours]


def contour_approx_poly_filter_quadrilater(contours):
    return filter_4_edges_contours(contours_to_poly(contours))
    # convex_contours = [cv2.convexHull(contour) for contour in contours]


# See https://docs.opencv.org/4.8.0/d1/d32/tutorial_py_contour_properties.html
def calc_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
    return solidity


def filter_4_edges_contours(contours):
    return [c for c in contours if len(c) is 4]


def filter_contours_by_size_solidity(contours, min_solidity: float = 0.9):
    return [
        contour
        for contour in contours
        if cv2.contourArea(contour) > 100 and calc_solidity(contour) > min_solidity
    ]
