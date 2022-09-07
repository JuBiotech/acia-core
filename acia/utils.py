"""Global utilities
"""

import numpy as np


def lut_mapping(image, in_min, in_max, out_min, out_max, dtype=None):
    mapped_data = (np.clip(image, in_min, in_max) - in_min) / (in_max - in_min) * (
        out_max - out_min
    ) + out_min

    if dtype:
        mapped_data = mapped_data.astype(dtype)

    return mapped_data


def pairwise_distances(points: np.ndarray):
    distances = []

    if len(points) == 0:
        return distances

    for a, b in zip(points, points[1:]):
        distances.append(np.linalg.norm(a - b))

    return distances
