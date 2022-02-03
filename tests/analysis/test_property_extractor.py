import unittest
from acia.analysis import (
    AreaEx,
    ExtractorExecutor,
    FluorescenceEx,
    FrameEx,
    IdEx,
    LengthEx,
    PositionEx,
    PropertyExtractor,
    TimeEx,
)
import pint
import numpy as np
from itertools import product, starmap

from acia.base import Contour, ImageSequenceSource, Overlay
from acia.segm.local import InMemorySequenceSource, LocalImage, LocalImageSource


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
        contours = [Contour([[0, 0], [1, 0], [1, 1], [0, 1]], -1, frame=0, id=23)]
        overlay = Overlay(contours)

        ureg = pint.UnitRegistry()

        image = np.zeros((200, 200))
        image[0, 0] = 2
        image[0, 1] = 5
        image[1, 0] = 6
        image[1, 1] = 10
        image_source = LocalImageSource.from_array(image)

        # test basic extractors
        df = ExtractorExecutor().execute(
            overlay=overlay,
            images=image_source,
            extractors=[
                IdEx(),
                FrameEx(),
                AreaEx(0.07 * ureg.micrometer**2),
                LengthEx(),
                TimeEx(input_unit="15 * minute"),  # one frame every 15 minutes
                PositionEx(input_unit=0.07 * ureg.micrometer),
                FluorescenceEx(channels=[0], channel_names=['gfp']),
                FluorescenceEx(channels=[0], channel_names=['gfp_mean'], summarize_operator=np.mean, parallel=1)
            ],
        )

        self.assertEqual(df["area"][0], 0.07)
        self.assertEqual(df["length"][0], 1)
        self.assertEqual(df["id"][0], 23)
        self.assertEqual(df["frame"][0], 0)
        self.assertEqual(df["time"][0], 0 * 15 / 60)
        self.assertEqual(df["position_x"][0], 0.5 * 0.07)
        self.assertEqual(df["position_y"][0], 0.5 * 0.07)
        self.assertEqual(df["gfp"][0], 5.5)
        self.assertEqual(df["gfp_mean"][0], np.mean([2, 5, 6, 10]))

    def test_parallel_fluorescence_extraction(self):
        squared_num = 30
        contours = [Contour([[0, 0], [1, 0], [1, 1], [0, 1]], -1, frame=0, id=23) for id,frame in product(list(range(squared_num)), list(range(squared_num)))]
        overlay = Overlay(contours)

        image = np.zeros((200, 200))
        image[0, 0] = 2
        image[0, 1] = 5
        image[1, 0] = 6
        image[1, 1] = 10
        image_sources =  InMemorySequenceSource(np.stack([image] * squared_num))

        # test basic extractors
        df = ExtractorExecutor().execute(
            overlay=overlay,
            images=image_sources,
            extractors=[
                FluorescenceEx(channels=[0], channel_names=['fl1']),
                FluorescenceEx(channels=[0], channel_names=['fl1_mean'], summarize_operator=np.mean),
            ],
        )

        np.testing.assert_array_equal(df['fl1'], [5.5] * len(df))
        np.testing.assert_array_equal(df['fl1_mean'], [np.mean([2, 5, 6, 10])] * len(df))


if __name__ == "__main__":
    unittest.main()
