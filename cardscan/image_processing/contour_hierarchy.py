from typing import List


class Hierarchy:
    NEXT = 0
    PREVIOUS = 1
    FIRST_CHILD = 2
    PARENT = 3
    OUTER = 0
    INNER = 1
    BLACK_OUTER = WHITE_INNER = 1
    WHITE_OUTER = BLACK_INNER = 0

    def __init__(self, h):
        self.h = h[0]

    def black_outer_entries(self):
        entries = []
        for child in self.siblings_from_first(0):
            for grandchild in self.children(child):
                entries.append(grandchild)
        return entries

    def next(self, i):
        return self.h[i][self.NEXT]

    def parent(self, i):
        return self.h[i][self.PARENT]

    def first_child(self, i: int):
        return self.h[i][self.FIRST_CHILD]

    def siblings_from_first(self, i: int) -> List[int]:
        siblings = []
        while i != -1:
            siblings.append(i)
            i = self.next(i)
        return siblings

    def children(self, i: int) -> List[int]:
        first_child = self.first_child(i)
        return self.siblings_from_first(first_child)

    def grandchildren(self, i: int) -> List[int]:
        grandchildren = []
        for child in self.children(i):
            for grandchild in self.children(child):
                grandchildren.append(grandchild)
        return grandchildren


def _aggreate_inner_unique_contours(
    hier: Hierarchy, outer_contours: List[int], aggregated_results=[]
):
    for outer_contour in outer_contours:
        inner_contours = hier.children(outer_contour)
        child_count = len(inner_contours)
        if child_count == 1:
            aggregated_results.append(inner_contours[0])
        elif child_count:
            nested_outer_contours = []
            for inner_contour in inner_contours:
                nested_outer_contours += hier.children(inner_contour)
            _aggreate_inner_unique_contours(
                hier, nested_outer_contours, aggregated_results
            )


def inner_unique_contours(h, c):
    if h is None or len(c) == 0:
        return []
    hier = Hierarchy(h)
    i_results = []
    _aggreate_inner_unique_contours(hier, hier.black_outer_entries(), i_results)
    contour_results = [c[i] for i in i_results]
    return contour_results
