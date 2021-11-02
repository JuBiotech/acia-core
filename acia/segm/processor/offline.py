from acia.base import Contour, ImageSequenceSource, Overlay, Processor
from mmdet.apis import init_detector
from mmcv.runner import wrap_fp16_model


from .predict import contour_from_mask
from .predict import prediction

import tqdm
import numpy as np


class OfflineModel(Processor):
    '''
        Model that runs on the local computer
    '''
    def __init__(self, config_file, parameter_file, half=False, device='cuda', tiling=None):
        '''
            config_file: model configuration file
            parameter_file: model checkpoint file
            half: enables half-precision (16-bit) execution. A bit faster.
            device: chooses the device to execute (e.g. 'cpu' or 'cuda' or 'cuda:0')
        '''
        # store file destinations
        self.config_file = config_file
        self.parameter_file = parameter_file
        # empty model instance
        self.model = None
        # half-precision execution
        self.half = half
        # determine the device
        self.device = device

        self.tiling = tiling

    def load_model(self, device=None, cfg_options=None, half=False):
        '''
            Load model from definitions

            device: device type, e.g. 'cpu' or 'cuda'
            cfg_options: overwrite configuration options e.g. {'test_cfg.rpn.nms_thr': 0.7}
        '''
        # init model
        if self.model is None:
            self.model = init_detector(self.config_file, self.parameter_file, device='cuda', cfg_options=cfg_options)
            if half:
                # make it 16-bit
                wrap_fp16_model(self.model)

        return self.model

    def predict(self, source: ImageSequenceSource) -> Overlay:
        '''
            Predicts the overlay for an image sequence

            source: image sequence source
            tiling: whether to enable tiling
        '''
        self.load_model(half=self.half, device=self.device)#, cfg_options={'test_cfg.rcnn.nms.iou_threshold': 0.3, 'test_cfg.rcnn.score_thr': 0.5})

        # TODO: super strange without [] it takes some other list as initialization. This leads to detected cells from other images...
        overlay = Overlay([])

        for frame_id, image in tqdm.tqdm(enumerate(source)):

            pred_result = prediction(image, self.model, tiling=self.tiling)

            if len(pred_result) == 0:
                # no predictions
                continue

            all_masks = np.stack([det['mask'] for det in pred_result])
            all_contours = [contour_from_mask(mask, 0.5) for mask in all_masks]
            # drop non-sense contours
            all_contours = list(filter(lambda comb: len(comb[1]) >= 5, zip(pred_result, all_contours)))

            contours = [Contour(cont, pred['score'], frame_id, id=-1) for pred,cont in all_contours]
            overlay.add_contours(contours)

        return overlay
