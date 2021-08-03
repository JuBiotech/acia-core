from typing import List


class Contour:
    def __init__(self, coordinates, score, frame):
        self.coordinates = coordinates
        self.score = score
        self.frame = frame


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

    def timeIterator(self, startFrame=0, endFrame=None):
        '''
            Creates an iterator that returns an Overlay for every frame between starFrame and endFrame

            startFrame: first frame number
            endFrame: last frame number
        '''
        assert startFrame >= 0
        if endFrame is None:
            # automatically determine the max endTime
            endFrame = max([c.frame for c in self.contours])
        else:
            assert endFrame >= 0

        # iterate frames
        for frame in range(startFrame, endFrame+1):
            # filter sub overlay with all contours in the frame
            yield Overlay(list(filter(lambda contour: contour.frame == frame, self.overlay)))


class Processor(object):
    pass


class ImageSequenceSource(object):
    pass
