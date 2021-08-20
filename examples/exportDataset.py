from acia.segm.output import CocoDataset
from acia.base import ImageRoISource
from typing import List
from omero.gateway import BlitzGateway
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
import getpass
import cv2
import numpy as np
import omero

import logging

def has_all_tags(object, tag_list: List[str] = []):
    tag_list = tag_list.copy()
    for ann in object.listAnnotations():
        if ann.OMERO_TYPE == omero.model.TagAnnotationI:
            if ann.getTextValue() in tag_list:
                del tag_list[tag_list.index(ann.getTextValue())]

        if len(tag_list) == 0:
            break

    if len(tag_list) == 0:
        return True
    else:
        return False

if __name__ == '__main__':
    serverUrl = 'ibt056'
    username = 'root'
    password = getpass.getpass(f'Password for {username}@{serverUrl}: ')

    omero_cred = {
        'username': username,
        'serverUrl': serverUrl,
        'password': password
    }

    logging.info("Connect to omero...")
    with BlitzGateway(username, password, host=serverUrl, port=4064, secure=True) as conn:
        tags = conn.getObjects("TagAnnotation")

        print('Available tags:')
        for tag in tags:
            print("\t", tag.textValue)

        print("Datasets with 'C. glutamicum': ")
        dataset_list = []
        for dataset in conn.getObjects("Dataset"):
            if has_all_tags(dataset, ['C. glutamicum']):
                print("\t", dataset.getName())
                dataset_list.append(dataset.getId())

        print("Ground truth data for C. glutamicum")
        image_list = []
        for dataset_id in dataset_list:
            for image in conn.getObjects("Image", opts={'dataset': dataset_id}):
                if has_all_tags(image, ['gold-standard']):
                    dataset = image.getParent()
                    project = dataset.getParent()
                    print(f'{project.getName()} > {dataset.getName()} > {image.getName()}')

                    # add the image to the list
                    image_list.append(image.getId())

        # combine all the image sequences with their rois
        irs_list = []
        for image_id in image_list:
            irs = ImageRoISource(
                OmeroSequenceSource(imageId = image_id, **omero_cred),
                OmeroRoISource(image_id, **omero_cred)
            )
            irs_list.append(irs)

        cd = CocoDataset()
        cd.add(irs_list)
        cd.write('coco')


    exit(1)