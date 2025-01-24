"""Omnipose segmentation implementation"""

from pathlib import Path

import torch
from cellpose_omni import models

from acia.base import ImageSequenceSource, Overlay
from acia.segm.formats import overlay_from_masks

from . import SegmentationProcessor


class OmniposeSegmenter(SegmentationProcessor):
    """Omnipose segmentation implementation"""

    def __init__(self, use_GPU: bool = None, model="bact_phase_omni"):
        if use_GPU is None:
            use_GPU = torch.cuda.is_available()
        self.use_GPU = use_GPU

        model_type = None
        model_path = None

        if Path(model).exists() and Path(model).is_file():
            model_path = model
        elif model in models.MODEL_NAMES:
            model_type = model
        else:
            raise ValueError(
                "Specified model is neither predefined nor a url to download"
            )

        if model_type:
            self.model = models.CellposeModel(gpu=use_GPU, model_type=model_type)
        if model_path:
            self.model = models.CellposeModel(
                gpu=use_GPU, pretrained_model=model_path, nclasses=3, nchan=2
            )

    @staticmethod
    def __predict(images, model):
        chans = [0, 0]  # this means segment based on first channel, no second channel

        # define parameters
        mask_threshold = -1
        verbose = 0  # turn on if you want to see more output
        transparency = True  # transparency in flow output
        rescale = (
            None  # give this a number if you need to upscale or downscale your images
        )
        omni = True  # we can turn off Omnipose mask reconstruction, not advised
        flow_threshold = 0.4  # default is .4, but only needed if there are spurious masks to clean up; slows down output
        resample = (
            True  # whether or not to run dynamics on rescaled grid or original grid
        )
        cluster = True  # use DBSCAN clustering

        masks, flows, styles = model.eval(
            images,
            channels=chans,
            rescale=rescale,
            mask_threshold=mask_threshold,
            transparency=transparency,
            flow_threshold=flow_threshold,
            omni=omni,
            cluster=cluster,
            resample=resample,
            verbose=verbose,
        )

        return masks, flows, styles

    def predict(self, images: ImageSequenceSource) -> Overlay:
        return self(images)

    def __call__(self, images: ImageSequenceSource) -> Overlay:

        imgs = []
        for image in images:
            raw_image = image.raw

            imgs.append(raw_image)

        masks, _, _ = self.__predict(imgs, self.model)

        return overlay_from_masks(masks)
