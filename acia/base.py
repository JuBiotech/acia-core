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

class Processor(object):
    pass

class ImageSequenceSource(object):
    pass