import unittest
from acia.base import Contour
import numpy as np


class TestContour(unittest.TestCase):

    def test_center(self):
        contour = [[0, 0], [1, 0], [1, 1], [0, 1]]

        # simple contour
        np.testing.assert_array_equal(Contour(contour, 0, 0, 0).center, np.array([0.5, 0.5], dtype=np.float32))

        # unequal point sampling
        contour = [[0, 0], [0.5, 0], [1, 0], [1, 1], [0, 1]]
        np.testing.assert_array_equal(Contour(contour, 0, 0, 0).center, np.array([0.5, 0.5], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
