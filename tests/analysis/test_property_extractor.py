import unittest
from acia.analysis import (
    AreaEx,
    ExtractorExecutor,
    FrameEx,
    IdEx,
    LengthEx,
    PositionEx,
    PropertyExtractor,
    TimeEx,
)
import pint

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
        contours = [Contour([[0, 0], [1, 0], [1, 1], [0, 1]], -1, frame=5, id=23)]
        overlay = Overlay(contours)

        ureg = pint.UnitRegistry()

        # test basic extractors
        df = ExtractorExecutor().execute(
            overlay=overlay,
            extractors=[
                IdEx(),
                FrameEx(),
                AreaEx(0.07 * ureg.micrometer**2),
                LengthEx(),
                TimeEx(input_unit="15 * minute"),  # one frame every 15 minutes
                PositionEx(input_unit=0.07 * ureg.micrometer),
            ],
            images=None,
        )

        self.assertEqual(df["area"][0], 0.07)
        self.assertEqual(df["length"][0], 1)
        self.assertEqual(df["id"][0], 23)
        self.assertEqual(df["frame"][0], 5)
        self.assertEqual(df["time"][0], 5 * 15 / 60)
        self.assertEqual(df["position_x"][0], 0.5 * 0.07)
        self.assertEqual(df["position_y"][0], 0.5 * 0.07)


if __name__ == "__main__":
    unittest.main()
