from __future__ import annotations

from typing import Callable, Iterator, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import tqdm
from functools import partial

def unpack(data, function):
    return function(*data)

class Contour:
    def __init__(self, coordinates, score, frame, id):
        self.coordinates = coordinates
        self.score = score
        self.frame = frame
        self.id = id

    '''
        Render contour mask onto existing image

        img: pillow image
        fillValue: mask values inside the contour
        outlineValues: mask values on the outline (border)
    '''
    def _toMask(self, img, maskValue=1, outlineValue=1, draw=None):
        if draw is None:
            draw = ImageDraw.Draw(img)
        draw.polygon(self.coordinates, outline=outlineValue, fill=maskValue)
        mask = np.array(img, np.bool)

        return mask

    '''
        Render contour mask onto new image

        height: height of the image
        width: width of the image
        fillValue: mask values inside the contour
        outlineValues: mask values on the outline (border)
    '''
    def toMask(self, height, width, fillValue=1, outlineValue=1):
        img = Image.new('L', (width, height), 0)

        return self._toMask(img, maskValue=fillValue, outlineValue=outlineValue)

    def draw(self, image, draw=None, outlineColor=(255, 255, 0), fillColor=None):
        if draw is None:
            draw = ImageDraw.Draw(image)
        draw.polygon(self.coordinates, outline=outlineColor, fill=fillColor)


class Overlay:
    def __init__(self, contours: List[Contour] = []):
        self.contours = contours

    def add_contour(self, contour: Contour):
        self.contours.append(contour)

    def add_contours(self, contours: List[Contour]):
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
        return np.unique([c.frame for c in self.contours])

    def timeIterator(self, startFrame=None, endFrame=None):
        '''
            Creates an iterator that returns an Overlay for every frame between starFrame and endFrame

            startFrame: first frame number
            endFrame: last frame number
        '''
        if startFrame is None:
            startFrame = np.min(self.frames())

        if endFrame is None:
            endFrame = np.max(self.frames())

        assert startFrame >= 0
        assert endFrame >= 0
        assert endFrame <= np.max(self.frames())

        # iterate frames
        for frame in range(startFrame, endFrame+1):
            # filter sub overlay with all contours in the frame
            yield Overlay(list(filter(lambda contour: contour.frame == frame, self.contours)))

    '''
        Turn the individual overlays into masks. For every time point we create a mask of all contours.

        returns: List of masks (np.array[bool])

        height: height of the image
        width: width of the image
    '''
    def toMasks(self, height, width) -> List[np.array]:
        masks = []
        for timeOverlay in self.timeIterator():
            img = Image.new('L', (width, height), 0)
            for cont in timeOverlay:
                cont._toMask(img, maskValue=1, outlineValue=1)
            mask = np.array(img, np.bool)
            masks.append(mask)

        return masks

    def draw(self, image, outlineColor: str | Callable[[Contour], Tuple[int]] = None, fillColor: str | Callable[[Contour], Tuple[int]] = None):
        for timeOverlay in self.timeIterator():
            for cont in timeOverlay:
                oc_local = outlineColor
                fc_local = fillColor

                if oc_local and isinstance(oc_local, Callable):
                    oc_local = oc_local(cont)
                if fc_local and isinstance(fc_local, Callable):
                    fc_local = fc_local(cont)

                cont.draw(image, outlineColor=oc_local, fillColor=fc_local)


class Processor(object):
    pass


class ImageSequenceSource(object):
    pass

class RoISource(object):
    pass

class ImageRoISource(object):
    '''
        Contains both, the image and the RoI Source. Provides a joint iterator
    '''
    def __init__(self, imageSource: ImageSequenceSource, roiSource: RoISource):
        self.imageSource = imageSource
        self.roiSource = roiSource

    def __iter__(self) -> Iterator[Tuple[np.array, Overlay]]:
        return zip(iter(self.imageSource), iter(self.roiSource))

    def __len__(self):
        return min(len(self.imageSource), len(self.roiSource))

    def apply_parallel(self, function, num_workers=None):
        import multiprocessing
        from tqdm.contrib.concurrent import process_map
        if num_workers is None:
            num_workers = int(np.floor(multiprocessing.cpu_count()*2/3))

        def limit():
            for i, el in enumerate(self):
                yield el

        return process_map(function, self, max_workers=num_workers, chunksize=4)

    def apply_parallel_star(self, function, num_workers=None):
        import multiprocessing
        from tqdm.contrib.concurrent import process_map
        if num_workers is None:
            num_workers = int(np.floor(multiprocessing.cpu_count()*2/3))

        return process_map(partial(unpack, function=function), self, max_workers=num_workers, chunksize=4)

    def apply(self, function):
        def limit():
            for i, el in enumerate(self):
                yield el
                #if i == 11:
                #    break

        return list(tqdm.tqdm(map(function, limit())))
