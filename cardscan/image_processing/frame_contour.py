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


def filter_contours_by_size(contours, min_size: int = 2000):
    contour_map = {}

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > min_size:
            if calc_solidity(contours[i]) > 0.8:
                contour_map[i] = contours[i]
                continue

    return contour_map


def filter_containing_contours(contour_map, hierarchy):
    """Unefficient algrorithm to filter the second most nested contours."""
    for i in list(contour_map.keys()):
        current_index = i
        while hierarchy[0][current_index][3] > 0:
            if hierarchy[0][current_index][3] in contour_map.keys():
                contour_map.pop(hierarchy[0][current_index][3])
            current_index = hierarchy[0][current_index][3]

    return list(contour_map.values())


## Comparison function for sorting contours
# def get_contour_precedence(contour, cols):
#    # USAGE: final_contour_list.sort(key=lambda x: get_contour_precedence(x, binary.shape[1]))
#    tolerance_factor = 200
#    origin = cv2.boundingRect(contour)
#    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
