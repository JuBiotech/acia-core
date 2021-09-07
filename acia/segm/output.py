from __future__ import annotations


from typing import Iterable, List
from acia.base import ImageRoISource, Overlay
import os
import json
import datetime
import shutil
import tqdm
import skimage.io
from pycococreatortools import pycococreatortools
import numpy as np
import cv2

import logging

def drawJointMask(image_id: int, height: int, width: int, overlay: Overlay):
    joint_mask = np.zeros((height, width), dtype=np.uint8)
    annotations = []

    local_seg_index = 0

    for contour in overlay:
        mask = np.zeros((height, width), dtype=np.uint8)

        try:
            contour = np.array(contour.coordinates)

            # we need the contour mask as some contour points might lie outside of the image bondaries (they're simply neglected)
            contour_mask = np.all(contour > 0, axis=1) & (contour[:,0] < width) & (contour[:,1] < height)

            # update the contour to only consist of points inside the image frame
            contour = contour[contour_mask]

            if len(contour) < 3:
                # a reasonable contour should consist of at least 3 points
                continue

        except Exception as e:
            print(e)
            print(contour)
            continue

        # draw the contour as a mask
        xy = np.array([contour]).astype(np.int32)
        cv2.fillPoly(mask, xy, 1, lineType=cv2.LINE_AA)

        joint_mask |= mask

        # add the roi as coco annotation
        category_info = {'id': 1, 'is_crowd': False}
        binary_mask = mask
        seg_index = int('%d%04d' % (image_id, local_seg_index))
        annotation_info = pycococreatortools.create_annotation_info(
                seg_index, image_id, category_info, binary_mask,
                (width, height), tolerance=0.0)

        if annotation_info is not None:
            annotations.append(annotation_info)

        local_seg_index += 1

    joint_mask = joint_mask.astype(np.uint8)

    return joint_mask, annotations


class CocoDataset:

    def __init__(self):
        self.sources = []

    def add(self, item: ImageRoISource | List[ImageRoISource]):
        if isinstance(item, Iterable):
            # iterable
            self.sources += item
        else:
            # not iterable
            self.sources.append(item)

    def write(self, base_folder: str, mode="train"):
        '''
        mode: 'train' | 'val'
        '''

        INFO = {
            "description": "CocoDataset - for cell segmentation",
            "url": "https://github.com/waspinator/pycococreator",
            "version": "0.1.0",
            "year": 2021,
            "contributor": "JojoDevel",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

        CATEGORIES = [
            {
                'id': 1,
                'name': 'cell',
                'supercategory': 'cell',
            },
            {
                'id': 0,
                'name': 'background',
                'supercategory': 'background'
            }
        ]

        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

        if mode == 'train':
            image_dir = os.path.join(base_folder, 'train2017')
            stuffthings_dir = os.path.join(base_folder, 'stuffthingmaps/train2017')
            annotation_file = os.path.join(base_folder, 'annotations/instances_train2017.json')
        else:
            image_dir = os.path.join(base_folder, 'val2017')
            stuffthings_dir = os.path.join(base_folder, 'stuffthingmaps/val2017')
            annotation_file = os.path.join(base_folder, 'annotations/instances_val2017.json')

        
        if os.path.exists(image_dir):
            # delete it to prevent misconcepted datasets
            logging.info('Delete existing image dir to prevent misconcepted datasets.')
            shutil.rmtree(image_dir)
        if os.path.exists(stuffthings_dir):
            logging.info('Delete existing stuffthingsmaps to prevent misconcepted datasets.')
            shutil.rmtree(stuffthings_dir)

        # create paths if needed
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(stuffthings_dir):
            os.makedirs(stuffthings_dir)
        os.makedirs(os.path.dirname(annotation_file), exist_ok=True)

        image_infos = []
        annotations = []

        for input_index, image_roi_source in enumerate(tqdm.tqdm(self.sources)):
            for image_index, (image, rois) in enumerate(tqdm.tqdm(image_roi_source)):
                if len(rois) == 0:
                    logging.info('No detections in a frame! -> Skip')
                    continue

                height, width = image.shape[:2]

                # store the image
                image_id = int('%04d%04d' % (input_index, image_index))
                image_filename = '%04d_%04d.png' % (input_index, image_index)
                image_path = os.path.join(image_dir, image_filename)
                skimage.io.imsave(image_path, image)

                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), (width, height))

                image_infos.append(image_info)

                joint_mask, local_annotations = drawJointMask(image_id, height, width, rois)
                annotations += local_annotations
                image_path = os.path.join(stuffthings_dir, image_filename)
                skimage.io.imsave(image_path, joint_mask, check_contrast=False)

        coco_output['images'] = image_infos
        coco_output['annotations'] = annotations

        with open(os.path.join(annotation_file), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

'''
    Wrapper for opencv video writer. Simplifies usage
'''
class VideoExporter:

    def __init__(self, filename, framerate):
        self.filename = filename
        self.framerate = framerate
        self.out = None
        self.frame_height = None
        self.frame_width = None

    def __del__(self):
        if self.out:
            self.close()

    def write(self, image):
        height, width = image.shape[:2]
        if self.out is None:
            self.frame_height, self.frame_width = image.shape[:2]
            self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc('M','J','P','G'), self.framerate, (self.frame_width, self.frame_height))
        if self.frame_height != height or self.frame_width != width:
            logging.warning('You add images of different resolution to the VideoExporter. This may cause problems (e.g. black video output)!')
        self.out.write(image)

    def close(self):
        if self.out:
            self.out.release()
            self.out = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
