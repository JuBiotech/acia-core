from __future__ import annotations

from typing import List, Optional
import pandas as pd

from acia.base import Overlay
from pint._typing import UnitLike
from pint import Quantity, Unit
import numpy as np
import numpy.ma as ma
from PIL import Image, ImageDraw

DEFAULT_UNIT_LENGTH = "micrometer"
DEFAULT_UNIT_AREA = "micrometer ** 2"


class PropertyExtractor(object):
    def __init__(
        self, name: str, input_unit: UnitLike, output_unit: Optional[UnitLike] = None
    ):
        self.name = name

        # try to parse input quantity
        self.input_unit = Quantity(input_unit)
        if self.input_unit.dimensionless and isinstance(
            self.input_unit.magnitude, Unit
        ):
            # if we have no dimension and magnitude is unit -> we better go with a unit
            self.input_unit = Unit(input_unit)
        if output_unit:
            # parse output unit
            self.output_unit = Unit(output_unit)
        else:
            # no conversion if no output unit is specified
            self.output_unit = self.input_unit

        # test the conversion here
        self.output_unit.is_compatible_with(self.input_unit)

    def extract(self, overlay: Overlay, images: List, df: pd.DataFrame):
        """Extract the desired properties for a single contour

        Args:
            contour (Contour): contour for the qunatity
            overlay (Overlay): overlay containing all contours
            df (pd.DataFrame): DataFrame of properties so far

        Raises:
            NotImplementedError: Please implement this method
        """
        raise NotImplementedError()

    def convert(self, input: float | Quantity) -> float:
        if isinstance(input, Quantity):
            # 1. convert input to input unit
            # 2. scale with input unit
            # 3. convert to output unit
            return (
                (input.to(self.input_unit).magnitude * self.input_unit)
                .to(self.output_unit)
                .magnitude
            )
        else:
            # 1. scale input with input unit/quantity
            # 2. convert to output unit
            return (input * self.input_unit).to(self.output_unit).magnitude


class ExtractorExecutor(object):
    def execute(self, overlay: Overlay, images: List, extractors: List[PropertyExtractor] = []):
        df = pd.DataFrame()
        for extractor in extractors:
            result_df = extractor.extract(overlay, df)

            df = pd.concat([df, result_df], ignore_index=False, sort=False, axis=1)

        return df


class AreaEx(PropertyExtractor):
    def __init__(
        self,
        input_unit: Optional[UnitLike] = DEFAULT_UNIT_AREA,
        output_unit: Optional[UnitLike] = DEFAULT_UNIT_AREA,
    ):
        PropertyExtractor.__init__(
            self, "area", input_unit=Unit(input_unit), output_unit=Unit(output_unit)
        )

    def extract(self, overlay: Overlay, df: pd.DataFrame):
        areas = []
        for cont in overlay:
            areas.append(self.convert(cont.area))

        return pd.DataFrame({self.name: areas})


class LengthEx(PropertyExtractor):
    def __init__(
        self,
        input_unit: Optional[UnitLike] = DEFAULT_UNIT_LENGTH,
        output_unit: Optional[UnitLike] = DEFAULT_UNIT_LENGTH,
    ):
        PropertyExtractor.__init__(
            self, "length", input_unit=Unit(input_unit), output_unit=Unit(output_unit)
        )

    @staticmethod
    def pairwise_distances(points):
        distances = []

        if len(points) == 0:
            return distances

        for a, b in zip(points, points[1:]):
            distances.append(np.linalg.norm(a - b))

        return distances

    def extract(self, overlay: Overlay, df: pd.DataFrame):
        lengths = []
        for cont in overlay:
            lengths.append(
                self.convert(
                    np.max(
                        LengthEx.pairwise_distances(
                            np.array(
                                cont.polygon.minimum_rotated_rectangle.exterior.coords
                            )
                        )
                    )
                )
            )

        return pd.DataFrame({self.name: lengths})


class FrameEx(PropertyExtractor):
    def __init__(self):
        super().__init__("frame", 1)

    def extract(self, overlay: Overlay, images: List, df: pd.DataFrame):
        frames = []
        for cont in overlay:
            frames.append(self.convert(cont.frame))

        return pd.DataFrame({self.name: frames})


class IdEx(PropertyExtractor):
    def __init__(self):
        super().__init__("id", 1)

    def extract(self, overlay: Overlay, images: List, df: pd.DataFrame):
        ids = []
        for cont in overlay:
            ids.append(self.convert(cont.id))
        return pd.DataFrame({self.name: ids})


class TimeEx(PropertyExtractor):
    def __init__(self, input_unit: UnitLike, output_unit: Optional[UnitLike] = "hour"):
        super().__init__("time", input_unit, output_unit)

    def extract(self, overlay: Overlay, images: List, df: pd.DataFrame):
        times = []
        for index, row in df.iterrows():
            times.append(self.convert(row["frame"]))

        return pd.DataFrame({self.name: times})


class FluorescenceEx(PropertyExtractor):
    def __init__(self, channels, channel_names, summarize_operator=np.median, input_unit: UnitLike = 1, output_unit: Optional[UnitLike] = 1):
        super().__init__("Fluorescence", input_unit=input_unit, output_unit=output_unit)

        assert len(self.channels) == self.channel_names, "Number of channels and number of channel names must comply"

        self.channels = channels
        self.channel_names = channel_names
        self.summarize_operator = summarize_operator

    def extract(self, overlay: Overlay, images: List, df: pd.DataFrame):
        channel_values = [] * len(self.channels)

        for cont in overlay:
            for ch_id, channel in enumerate(self.channels):
                image = images[cont.frame]
                raw_image = image.get_channel(channel)

                height, width = raw_image.shape[:2]
                img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(img)

                # draw cell mask
                roi_mask = cont._toMask(img, draw=draw)

                # create masked array
                masked_roi = ma.masked_array(image, mask=~roi_mask)

                # compute fluorescence response
                value = self.summarize_operator(masked_roi.compressed())

                channel_values[ch_id].append(value)

        return pd.DataFrame({self.channel_names[i]: channel_values[i] for i in range(len(self.channels))})
