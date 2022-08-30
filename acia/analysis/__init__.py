from __future__ import annotations
from functools import reduce
import logging
from pathlib import Path
import os
import shutil
import papermill as pm

from typing import List, Optional
import pandas as pd

from acia.base import BaseImage, ImageSequenceSource, Overlay
from pint._typing import UnitLike
from acia import Q_, U_
import numpy as np
import numpy.ma as ma
from PIL import Image, ImageDraw
from multiprocessing import Pool
from tqdm.auto import tqdm
from itertools import starmap


DEFAULT_UNIT_LENGTH = "micrometer"
DEFAULT_UNIT_AREA = "micrometer ** 2"


class PropertyExtractor(object):
    def __init__(
        self, name: str, input_unit: UnitLike, output_unit: Optional[UnitLike] = None
    ):
        self.name = name

        # try to parse input quantity
        self.input_unit = Q_(input_unit)
        if self.input_unit.dimensionless and isinstance(
            self.input_unit.magnitude, U_
        ):
            # if we have no dimension and magnitude is unit -> we better go with a unit
            self.input_unit = U_(input_unit)
        if output_unit:
            # parse output unit
            self.output_unit = U_(output_unit)
        else:
            # no conversion if no output unit is specified
            self.output_unit = self.input_unit

        # test the conversion here
        self.output_unit.is_compatible_with(self.input_unit)

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        """Extract the desired properties for a single contour

        Args:
            contour (Contour): contour for the qunatity
            overlay (Overlay): overlay containing all contours
            df (pd.DataFrame): DataFrame of properties so far

        Raises:
            NotImplementedError: Please implement this method
        """
        raise NotImplementedError()

    def convert(self, input: float | Q_) -> float:
        """ Converts input to the specified output unit

        Args:
            input (float | Quantity): Input value

        Returns:
            float: the magnitude in the output unit
        """
        if isinstance(input, Q_):
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
    def __init__(self) -> None:
        self.units = {}

    def execute(self, overlay: Overlay, images: List, extractors: List[PropertyExtractor] = []):
        df = pd.DataFrame()
        for extractor in tqdm(extractors):
            print(f"Extracting: {extractor.name}...")
            result_df, units = extractor.extract(overlay, images, df)

            df = pd.concat([df, result_df], ignore_index=False, sort=False, axis=1)

            self.units.update(**units)

        return df


class AreaEx(PropertyExtractor):
    def __init__(
        self,
        input_unit: Optional[UnitLike] = DEFAULT_UNIT_AREA,
        output_unit: Optional[UnitLike] = DEFAULT_UNIT_AREA,
    ):
        PropertyExtractor.__init__(
            self, "area", input_unit=input_unit, output_unit=output_unit
        )

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        areas = []
        for cont in overlay:
            areas.append(self.convert(cont.area))

        return pd.DataFrame({self.name: areas}), {self.name: self.output_unit}


class LengthEx(PropertyExtractor):
    """ Extracts width of cells based on the shorter edge of a minimum rotated bbox approximation"""
    def __init__(
        self,
        input_unit: Optional[UnitLike] = DEFAULT_UNIT_LENGTH,
        output_unit: Optional[UnitLike] = DEFAULT_UNIT_LENGTH,
    ):
        PropertyExtractor.__init__(
            self, "length", input_unit=input_unit, output_unit=output_unit
        )

    @staticmethod
    def pairwise_distances(points):
        distances = []

        if len(points) == 0:
            return distances

        for a, b in zip(points, points[1:]):
            distances.append(np.linalg.norm(a - b))

        return distances

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        lengths = []
        for cont in overlay:
            lengths.append(
                self.convert(
                    # longer edge of minimum roated bbox
                    np.max(
                        LengthEx.pairwise_distances(
                            np.array(
                                cont.polygon.minimum_rotated_rectangle.exterior.coords
                            )
                        )
                    )
                )
            )

        return pd.DataFrame({self.name: lengths}), {self.name: self.output_unit}


class WidthEx(PropertyExtractor):
    """ Extracts width of cells based on the shorter edge of a minimum rotated bbox approximation"""
    def __init__(
        self,
        input_unit: Optional[UnitLike] = DEFAULT_UNIT_LENGTH,
        output_unit: Optional[UnitLike] = DEFAULT_UNIT_LENGTH,
    ):
        PropertyExtractor.__init__(
            self, "width", input_unit=input_unit, output_unit=output_unit
        )

    @staticmethod
    def pairwise_distances(points):
        distances = []

        if len(points) == 0:
            return distances

        for a, b in zip(points, points[1:]):
            distances.append(np.linalg.norm(a - b))

        return distances

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        """ Extract width information for all contours"""
        widths = []
        for cont in overlay:
            widths.append(
                self.convert(
                    # shorter edge of bbox approximation
                    np.min(
                        # measure edge lengths of bbox approximation
                        WidthEx.pairwise_distances(
                            np.array(
                                # bbox approaximation
                                cont.polygon.minimum_rotated_rectangle.exterior.coords
                            )
                        )
                    )
                )
            )

        return pd.DataFrame({self.name: widths}), {self.name: self.output_unit}


class FrameEx(PropertyExtractor):
    def __init__(self):
        super().__init__("frame", 1)

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        frames = []
        for cont in overlay:
            frames.append(self.convert(cont.frame))

        return pd.DataFrame({self.name: frames}), {self.name: self.output_unit}


class IdEx(PropertyExtractor):
    def __init__(self):
        super().__init__("id", 1)

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        ids = []
        for cont in overlay:
            ids.append(self.convert(cont.id))
        return pd.DataFrame({self.name: ids}), {self.name: self.output_unit}


class TimeEx(PropertyExtractor):
    def __init__(self, input_unit: UnitLike, output_unit: Optional[UnitLike] = "hour"):
        super().__init__("time", input_unit, output_unit)

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        times = []
        for index, row in df.iterrows():
            times.append(self.convert(row["frame"]))

        return pd.DataFrame({self.name: times}), {self.name: self.output_unit}


class PositionEx(PropertyExtractor):
    def __init__(self, input_unit: UnitLike, output_unit: Optional[UnitLike] = DEFAULT_UNIT_LENGTH):
        super().__init__("position", input_unit=input_unit, output_unit=output_unit)

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        positions_x = []
        positions_y = []
        for cont in overlay:
            positions_x.append(self.convert(cont.center[0]))
            positions_y.append(self.convert(cont.center[1]))

        return pd.DataFrame({"position_x": positions_x, "position_y": positions_y}), {"position_x": self.output_unit, "position_y": self.output_unit}


class FluorescenceEx(PropertyExtractor):
    def __init__(self, channels, channel_names, summarize_operator=np.median, input_unit: UnitLike = '1', output_unit: Optional[UnitLike] = '', parallel=6):
        super().__init__("Fluorescence", input_unit=input_unit, output_unit=output_unit)

        self.channels = channels
        self.channel_names = channel_names
        self.summarize_operator = summarize_operator
        self.parallel = parallel

        assert len(self.channels) == len(self.channel_names), "Number of channels and number of channel names must comply"

    @staticmethod
    def extract_fluorescence(overlay: Overlay, image: BaseImage, channels: List[int], channel_names: List[str], summarize_operator):
        """Extract fluorescence information based on an overlay(segmentation) and corresponding image.

        Args:
            overlay (Overlay): Ovleray providing the image segmentation information
            image (BaseImage): the image itself
            channels (List[int]): list of channels (image channels) we want to investigate
            channel_names (List[str]): list of names for the channel results
            summarize_operator (_type_): summarizing operator, e.g. np.media, to compress all fluorescence values to a single one

        Returns:
            pd.DataFrame: pandas data frame containing columns of channel_names and the rows represent the extracted fluorescence
        """
        channel_values = [[] for _ in channels]

        for cont in overlay:
            for ch_id, channel in enumerate(channels):
                raw_image = image.get_channel(channel)

                height, width = raw_image.shape[:2]
                img = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(img)

                # draw cell mask
                roi_mask = cont._toMask(img, draw=draw)

                # create masked array
                masked_roi = ma.masked_array(raw_image, mask=~roi_mask)

                # compute fluorescence response
                value = summarize_operator(masked_roi.compressed())

                channel_values[ch_id].append(value)

        return pd.DataFrame({channel_names[i]: channel_values[i] for i in range(len(channels))})

    def extract(self, overlay: Overlay, images: ImageSequenceSource, df: pd.DataFrame):
        assert overlay.numFrames() == len(images), "Please make sure that the frames in your overlay fit to the frames in your image source"

        def iterator(timeIterator):
            for i, overlay in enumerate(timeIterator):
                yield (overlay, images.get_frame(i), self.channels, self.channel_names, self.summarize_operator)

        result = None

        if self.parallel > 1:
            try:
                with Pool(self.parallel) as p:
                    result = p.starmap(FluorescenceEx.extract_fluorescence, iterator(overlay.timeIterator()), chunksize=5)

            except Exception as e:
                logging.error("Parallel fluorescence extraction failed! Please run with 'parallel=1' to investigate the error!")
                raise e

        else:
            result = starmap(FluorescenceEx.extract_fluorescence, iterator(overlay.timeIterator()))

        # concatenate all results
        result = reduce(lambda a, b: pd.concat([a, b], ignore_index=True), result)

        return result, {self.channel_names[i]: self.output_unit for i in range(len(self.channels))}


def scale(output_path: Path, analysis_script: Path, image_ids: List[int], additional_parameters=None):
    """Scale an analysis notebook to several image sequences

    **Hint:** the analysis script should only use absolute paths as the file is copied and executed in another folder.

    Args:
        output_path (Path): the general output path to the storage
        analysis_script (Path): the template script
        image_ids (List[int]): list of (OMERO) image sources
    """

    if additional_parameters is None:
        additional_parameters = {}

    experiment_executions = []

    for image_id in tqdm(image_ids):

        # path to the new notebook file
        # every execution should have its own folder to store local files
        output_file = output_path / f"execution_{image_id}" / "notebook.ipynb"

        # create the directory (should not exist) and copy file to that
        os.makedirs(Path(output_file).parent, exist_ok=False)
        shutil.copy(analysis_script, output_file)

        # parameters to integrate into notebook
        parameters = dict(
            storage_folder=str(output_file.parent.absolute()),
            image_id=image_id,
            **additional_parameters
        )

        # execute the notebook
        pm.execute_notebook(
            output_file,
            output_file,
            parameters=parameters,
            cwd=output_file.parent
        )

        # save experiment in list
        experiment_executions.append(dict(parameters=parameters, storage_folder=output_file.parent))

    return experiment_executions
