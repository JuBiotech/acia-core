""" All basic functionality for acia """

from __future__ import annotations

import copy
import logging
import multiprocessing
from functools import partial
from typing import Callable, Iterator

import numpy as np
import tqdm
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from tqdm.contrib.concurrent import process_map


def unpack(data, function):
    return function(*data)


class Contour:
    """Class for object contour detection (e.g. Cell object)"""

    def __init__(self, coordinates, score: float, frame: int, id, label=None):
        """Create Contour

        Args:
            coordinates ([type]): [description]
            score (float): segmentation score
            frame (int): frame index
            id (any): unique id
            label: class-defining label of the contour
        """
        self.coordinates = np.array(coordinates, dtype=np.float32)
        self.score = score
        self.frame = frame
        self.id = id
        self.label = label

    def _toMask(self, img, maskValue=1, outlineValue=1, draw=None):
        """
        Render contour mask onto existing image

        img: pillow image
        fillValue: mask values inside the contour
        outlineValues: mask values on the outline (border)
        """
        if draw is None:
            draw = ImageDraw.Draw(img)
        draw.polygon(self.coordinates, outline=outlineValue, fill=maskValue)
        mask = np.array(img, np.bool)

        return mask

    def toMask(self, height, width, fillValue=1, outlineValue=1):
        """
        Render contour mask onto new image

        height: height of the image
        width: width of the image
        fillValue: mask values inside the contour
        outlineValues: mask values on the outline (border)
        """
        img = Image.new("L", (width, height), 0)
        return self._toMask(img, maskValue=fillValue, outlineValue=outlineValue)

    def draw(self, image, draw=None, outlineColor=(255, 255, 0), fillColor=None):
        if draw is None:
            draw = ImageDraw.Draw(image)
        draw.polygon(self.coordinates, outline=outlineColor, fill=fillColor)

    def scale(self, scale: float):
        """Apply scale factor to contour coordinates

        Args:
            scale (float): the multplication factor
        """
        self.coordinates *= scale

    @property
    def center(self):
        return np.array(Polygon(self.coordinates).centroid, dtype=np.float32)

    @property
    def area(self) -> float:
        """Compute the area inside the contour

        Returns:
            [float]: area
        """
        return self.polygon.area

    @property
    def polygon(self) -> Polygon:
        return Polygon(self.coordinates)

    def __repr__(self) -> str:
        return self.id


class Overlay:
    """Overlay contains Contours at different frames and provides functionalities iterate and modify them"""

    def __init__(self, contours: list[Contour], frames=None):
        self.contours = contours
        if frames is not None:
            frames = sorted(list(frames))
        self.__frames = frames

    def add_contour(self, contour: Contour):
        self.contours.append(contour)

    def add_contours(self, contours: list[Contour]):
        for cont in contours:
            self.add_contour(cont)

    def __iter__(self):
        return iter(self.contours)

    def __add__(self, other):
        jointContours = self.contours + other.contours
        return Overlay(jointContours)

    def __len__(self):
        return len(self.contours)

    def numFrames(self):
        return len(self.frames())

    def frames(self):
        if self.__frames:
            return self.__frames
        else:
            return np.unique([c.frame for c in self.contours])

    def scale(self, scale: float):
        """Scale the contour with the specified scale factor

           Applies the scale factor to all coordinates individually

        Args:
            scale (float): [description]
        """
        for cont in self.contours:
            cont.scale(scale)

    def croppedContours(self, cropping_parameters: tuple[slice, slice]):
        y, x = cropping_parameters
        miny, maxy, minx, maxx = y.start, y.stop, x.start, x.stop

        crop_rectangle = Polygon(
            [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        )

        def __crop_function_filter(contour: Contour):
            try:
                return crop_rectangle.contains(Polygon(contour.coordinates))
            # TODO: more precise exception catching here!
            # pylint: disable=W0703
            except Exception:
                # if we have problems to convert to shapely polygon, we cannot include it
                logging.warning(
                    "Have to drop Polygon: It cannot be converted into a shapely Polygon."
                )
                return False

        for cont in filter(__crop_function_filter, self.contours):
            new_cont = copy.deepcopy(cont)
            new_cont.coordinates -= np.array([minx, miny])

            yield new_cont

    def timeIterator(self, startFrame=None, endFrame=None, frame_range=None):
        """
        Creates an iterator that returns an Overlay for every frame between starFrame and endFrame

        startFrame: first frame number
        endFrame: last frame number
        """
        if len(self.frames()) == 0:
            yield Overlay([])

        if startFrame is None:
            startFrame = np.min(self.frames())

        if endFrame is None:
            endFrame = np.max(self.frames())

        assert startFrame >= 0
        assert endFrame >= 0
        assert endFrame <= np.max(self.frames())

        it_frames = range(startFrame, endFrame + 1)

        if self.__frames:
            it_frames = sorted(self.__frames)

        # iterate frames
        for frame in it_frames:
            if frame_range and frame not in frame_range:
                continue
            # filter sub overlay with all contours in the frame
            yield Overlay(
                list(
                    filter(
                        partial(
                            lambda contour, frame: contour.frame == frame, frame=frame
                        ),
                        self.contours,
                    )
                )
            )

    def toMasks(self, height, width) -> list[np.array]:
        """
        Turn the individual overlays into masks. For every time point we create a mask of all contours.

        returns: List of masks (np.array[bool])

        height: height of the image
        width: width of the image
        """
        masks = []
        for timeOverlay in self.timeIterator():
            img = Image.new("L", (width, height), 0)
            for cont in timeOverlay:
                cont._toMask(img, maskValue=1, outlineValue=1)
            mask = np.array(img, np.bool)
            masks.append(mask)

        return masks

    def draw(
        self,
        image,
        outlineColor: str | Callable[[Contour], tuple[int]] = None,
        fillColor: str | Callable[[Contour], tuple[int]] = None,
    ):
        imdraw = ImageDraw.Draw(image)
        for timeOverlay in self.timeIterator():
            for cont in timeOverlay:
                oc_local = outlineColor
                fc_local = fillColor

                if oc_local and isinstance(oc_local, Callable):
                    oc_local = oc_local(cont)
                if fc_local and isinstance(fc_local, Callable):
                    fc_local = fc_local(cont)

                cont.draw(image, outlineColor=oc_local, fillColor=fc_local, draw=imdraw)


class BaseImage:
    """Base class for an image from an image source"""

    @property
    def raw(self):
        raise NotImplementedError("Please implement this function!")

    @property
    def num_channels(self):
        raise NotImplementedError()

    def get_channel(self, channel: int):
        raise NotImplementedError()


class Processor:
    """Base class for a processor"""


class ImageSequenceSource:
    """Base class for an image sequence source (e.g. Tiff, OMERO, png, ...)"""

    @property
    def num_channels(self) -> int:
        raise NotImplementedError()

    def get_frame(self, frame: int) -> BaseImage:
        raise NotImplementedError()


class RoISource:
    """Base class for a RoI source (e.g. tiff metadata, OMERO, json, ...)"""


class ImageRoISource:
    """
    Contains both, the image and the RoI Source. Provides a joint iterator
    """

    def __init__(self, imageSource: ImageSequenceSource, roiSource: RoISource):
        self.imageSource = imageSource
        self.roiSource = roiSource

    def __iter__(self) -> Iterator[tuple[np.array, Overlay]]:
        return zip(iter(self.imageSource), iter(self.roiSource))

    def __len__(self):
        return min(len(self.imageSource), len(self.roiSource))

    def apply_parallel(self, function, num_workers=None):
        if num_workers is None:
            num_workers = int(np.floor(multiprocessing.cpu_count() * 2 / 3))

        return process_map(function, self, max_workers=num_workers, chunksize=4)

    def apply_parallel_star(self, function, num_workers=None):
        if num_workers is None:
            num_workers = int(np.floor(multiprocessing.cpu_count() * 2 / 3))

        return process_map(
            partial(unpack, function=function),
            self,
            max_workers=num_workers,
            chunksize=4,
        )

    def apply(self, function):
        def limit():
            for _, el in enumerate(self):
                yield el

        return list(tqdm.tqdm(map(function, limit())))

    def apply_star(self, function):
        return list(tqdm.tqdm(map(partial(unpack, function=function), self)))
