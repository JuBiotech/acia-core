import numpy as np
import tifffile
import cv2
import os.path as osp
import logging

from acia.base import ImageSequenceSource, Overlay, Contour, RoISource
import roifile

def prepare_image(image, normalize_image=True):
    """Normalize and convert image to RGB.

    Args:
        image ([type]): [description]
        normalize_image (bool, optional): Whether to normalize the image into uint8 domain (0-255). Defaults to True.
    Returns:
        [np.array]: RGB image (Width, height, 3 color channels)
    """
    # normalize image space
    if normalize_image:
        min_val = np.min(image)
        max_val = np.max(image)
        image = np.floor((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    if len(image.shape) > 2:
        # select only the first channel
        image = image[0]

    if len(image.shape) == 2:
        # make it artificially rgb
        image = np.repeat(image[:, :, None], 3, axis=-1)

    return image

class LocalImageSource(ImageSequenceSource):
    def __init__(self, file_path: str, normalize_image=True):
        self.file_path = file_path
        self.normalize_image = normalize_image

        if not osp.isfile(self.file_path):
            logging.warning(f'File "{self.file_path}" does not exist!')

    def __iter__(self):
        image = cv2.imread(self.file_path)

        yield prepare_image(image, self.normalize_image)

    def __len__(self):
        return 1

class LocalSequenceSource(ImageSequenceSource):
    def __init__(self, tif_file, normalize_image=True):
        self.filename = tif_file
        self.normalize_image = normalize_image

    def __iter__(self):
        images = tifffile.imread(self.filename)

        for image in images:
            image = prepare_image(image, self.normalize_image)

            yield image

    def slice(self, start, end):
        images = tifffile.imread(self.filename)

        for image in images[start:end]:
            # normalize image space
            if self.normalize_image:
                min_val = np.min(image)
                max_val = np.max(image)
                image = np.floor((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            if len(image.shape) > 2:
                # select only the first channel
                image = image[0]

            if len(image.shape) == 2:
                # make it artificially rgb
                image = np.repeat(image[:, :, None], 3, axis=-1)

            yield image

class ImageJRoISource(RoISource):
    def __init__(self, filename, range=None):
        self.overlay = RoiStorer.load(filename)
        self.range = range

    def __iter__(self):
        return self.overlay.timeIterator(frame_range=self.range)


    def __len__(self) -> int:
            if self.range:
                min(len(self.overlay), len(self.range))
            return len(self.overlay)


class RoiStorer:
    '''
        Stores and loads overlay results in the roi format (readable by ImageJ)
    '''

    @staticmethod
    def store(overlay: Overlay, filename: str, append=False):
        '''
            Stores overlay results in the roi format (readable by fiji)

            overlay: the overlay to store
            filename: filename of the roi collection (e.g. rois.zip)
            append: appends the rois if the file already exists
        '''

        # generate imagej rois from the overlay
        rois = [roifile.ImagejRoi.frompoints(contour.coordinates, t=contour.frame) for contour in overlay]

        # remove existing file if necessary
        import os.path
        if not append and os.path.isfile(filename):
            os.remove(filename)

        # write them to file
        roifile.roiwrite(filename, rois)

    @staticmethod
    def load(filename: str):
        # read the imagej rois from file
        rois = roifile.roiread(filename)

        id = -1
        # convert them into contours (recover time position)
        contours = [Contour(roi.coordinates(), -1., roi.position-1, id=id) for roi in rois]

        # return the overlay
        return Overlay(contours)
