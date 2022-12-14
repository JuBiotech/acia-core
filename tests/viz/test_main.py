"""Module for testing visualization module"""

import unittest

import numpy as np

from acia.viz import draw_scale_bar, draw_time


class TestOverlays(unittest.TestCase):
    """Testing scalebar and time overlay"""

    def test_time_overlay(self):
        """Simply draw time overlay on empty image"""

        images = np.zeros((10, 200, 200, 3), dtype=np.uint8)

        for image in draw_time(images, (100, 100), "5 minutes"):
            self.assertTrue(isinstance(image, np.ndarray))
            np.testing.assert_array_equal(image.shape, (200, 200, 3))

    def test_time_overlay_with_bg(self):
        """Simply draw time overlay on empty image with background"""

        images = np.zeros((10, 200, 200, 3), dtype=np.uint8)

        for image in draw_time(
            images, (100, 100), "5 minutes", background_color=(0, 0, 0)
        ):
            self.assertTrue(isinstance(image, np.ndarray))
            np.testing.assert_array_equal(image.shape, (200, 200, 3))

    def test_scale_bar_overlay(self):
        """Simply draw scale bar on empty image"""

        images = np.zeros((10, 200, 200, 3), dtype=np.uint8)

        for image in draw_scale_bar(
            images,
            xy_position=(100, 100),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
        ):
            self.assertTrue(isinstance(image, np.ndarray))
            np.testing.assert_array_equal(image.shape, (200, 200, 3))

    def test_scale_bar_overlay_with_background(self):
        """Simply draw scale bar on empty image with a background color"""

        images = np.zeros((10, 200, 200, 3), dtype=np.uint8)

        for image in draw_scale_bar(
            images,
            xy_position=(100, 100),
            size_of_pixel="0.07 micrometer/pixel",
            bar_width="5 micrometer",
            bar_height="0.25 micrometer",
            background_color=(255, 0, 0),
        ):
            self.assertTrue(isinstance(image, np.ndarray))
            np.testing.assert_array_equal(image.shape, (200, 200, 3))


if __name__ == "__main__":
    unittest.main()
