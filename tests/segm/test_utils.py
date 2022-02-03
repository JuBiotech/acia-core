import unittest
from acia.segm.utils import compute_indices


class TestIndexing(unittest.TestCase):

    def test_both(self):

        setup = dict(
            size_t=4,
            size_z=4)

        self.assertEqual(compute_indices(0, **setup), (0, 0))
        self.assertEqual(compute_indices(1, **setup), (0, 1))
        self.assertEqual(compute_indices(8, **setup), (2, 0))
        self.assertEqual(compute_indices(3, **setup), (0, 3))
        self.assertEqual(compute_indices(10, **setup), (2, 2))

    def test_only_t(self):
        setup = dict(
            size_t=50,
            size_z=1
        )

        for i in range(setup['size_t']):
            self.assertEqual(compute_indices(i, **setup), (i, 0))

    def test_only_z(self):
        setup = dict(
            size_t=1,
            size_z=50
        )

        for i in range(setup['size_z']):
            self.assertEqual(compute_indices(i, **setup), (0, i))


if __name__ == '__main__':
    unittest.main()
