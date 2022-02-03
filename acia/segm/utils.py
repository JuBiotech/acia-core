from typing import Tuple
import numpy as np

def compute_indices(frame: int, size_t: int, size_z: int)-> Tuple[int, int]:
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
    else:
        raise ValueError("This state should not be reachable!")

    return t,z