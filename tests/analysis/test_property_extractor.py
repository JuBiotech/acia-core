import unittest
from acia.analysis import (
    AreaEx,
    ExtractorExecutor,
    FrameEx,
    IdEx,
    LengthEx,
    PropertyExtractor,
    TimeEx,
)

from acia.base import Contour, Overlay


class TestPropertExtractors(unittest.TestCase):

    def test_unit_conversion(self):
        # test basic conversion patterns

        self.assertAlmostEqual(
            PropertyExtractor("test", "meter", "millimeter").convert(1), 1000
        )
        self.assertAlmostEqual(
            PropertyExtractor("test", "micrometer", "millimeter").convert(1), 1e-3
        )
        self.assertAlmostEqual(
            PropertyExtractor("test", "liter", "milliliter").convert(1), 1000
        )
        self.assertAlmostEqual(
            PropertyExtractor("test", "micrometer", "micrometer").convert(1), 1
        )
        self.assertAlmostEqual(
            PropertyExtractor("test", "meter", "micrometer").convert(1), 1e6
        )

    def test_extractors(self):
        contours = [
            Contour([[0, 0], [1, 0], [1, 1], [0, 1]], -1, frame=5, id=23)
        ]
        overlay = Overlay(contours)

        # test basic extractors
        df = ExtractorExecutor().execute(
            overlay,
            [IdEx(), FrameEx(), AreaEx(), LengthEx(), TimeEx(input_unit="15 * minute")],  # one frame every 15 minutes
        )

        self.assertEqual(df["area"][0], 1)
        self.assertEqual(df["length"][0], 1)
        self.assertEqual(df["id"][0], 23)
        self.assertEqual(df["frame"][0], 5)
        self.assertEqual(df["time"][0], 5 * 15 / 60)


if __name__ == "__main__":
    unittest.main()
