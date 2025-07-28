"""Segmenter using YOLO: https://github.com/ultralytics/ultralytics"""

import logging

import numpy as np
from ultralytics import YOLO

from acia.base import ImageSequenceSource, Instance, Overlay


class YOLOSegmenter:
    """YOLOSegmenter using Yolo: https://github.com/ultralytics/ultralytics"""

    def __init__(self, model):

        # create CellPose model
        self.model = YOLO(model)

    def __call__(self, images: ImageSequenceSource, cellpose_params=None) -> Overlay:

        # list of images
        imgs = [im.raw for im in images]

        if len(imgs[0].shape) != 3 or imgs[0].shape[-1] != 3:
            logging.warning(
                "Wrong shape for YOLO images. They should have shape [H,W,3] but have %s",
                {imgs[0].shape},
            )

        # perform prediction using yolo
        results = self.model(imgs, retina_masks=True)

        # List of all instances
        instances = []

        # loop over all frames
        for frame, frame_data in enumerate(results):
            # loop over all masks
            for mask, box in zip(frame_data.masks, frame_data.boxes):

                # add one because they start couting at 0 (but zero is background in the mask)
                cls = int(box.cls.item()) + 1

                # get the mask data
                np_mask = (mask.data.cpu().numpy() * cls).astype(np.uint8)

                # create a new instance in the overlay
                instances.append(Instance(np_mask[0], frame, cls))

        ov = Overlay(instances)

        return ov
