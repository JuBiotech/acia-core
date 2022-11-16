"""Utils for segmentation data handling"""

from typing import Tuple

import numpy as np
from shapely.geometry import Polygon

from acia.base import Contour
from acia.utils import pairwise_distances


def compute_indices(frame: int, size_t: int, size_z: int) -> Tuple[int, int]:
    """Compute t and z values from a linearized frame number

    Args:
        frame (int): the linearized frame index
        size_t (int): the total size of the t dimension
        size_z (int): the total size of the z dimension

    Returns:
        (Tuple[int, int]): tuple of (t,z) indices
    """

    if size_t > 1 and size_z > 1:
        t = int(np.floor(frame / size_t))
        z = frame % size_t
    elif size_t > 1:
        t = frame
        z = 0
    elif size_z > 1:
        t = 0
        z = frame
    elif size_t == 1 and size_z == 1:
        t = 0
        z = 0
    else:
        raise ValueError("This state should not be reachable!")

    return t, z


def length_and_area(contour: Contour) -> Tuple[float, float]:
    """Compute length and area of a contour object (in pixel coordinates)

    Args:
        contour (Contour): contour object

    Returns:
        tuple[float, float]: length and area of the contour
    """

    polygon = Polygon(contour.coordinates)

    length = np.max(
        pairwise_distances(np.array(polygon.minimum_rotated_rectangle.exterior.coords))
    )
    return length, polygon.area
