from typing import Iterator, List, Tuple
import numpy as np
from PIL import Image, ImageDraw

class Contour:
    def __init__(self, coordinates, score, frame):
        self.coordinates = coordinates
        self.score = score
        self.frame = frame

    '''
        Render contour mask onto existing image

        img: pillow image
        fillValue: mask values inside the contour
        outlineValues: mask values on the outline (border)
    '''
    def _toMask(self, img, maskValue=1, outlineValue=1):
        ImageDraw.Draw(img).polygon(self.coordinates, outline=outlineValue, fill=maskValue)
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

        return self.__toMask(img, maskValue=fillValue, outlineValue=outlineValue)


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