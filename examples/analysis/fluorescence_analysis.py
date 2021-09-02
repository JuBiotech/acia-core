from acia.segm.output import VideoExporter
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
from acia.base import ImageRoISource
import cv2
import numpy as np
import os.path as osp
import tqdm

if __name__ == '__main__':
    # omero id of the image (Image ID)
    image_id = 470
    # fluorescence channels you want to monitor (usually 1 is the phase contrast)
    fluorescence_channels = [2, 3]

    # your user credentials
    credentials = dict(
        username='root',
        password='omero',
        serverUrl='ibt056',
    )

    # combine images and rois
    irs = ImageRoISource(
        OmeroSequenceSource(image_id, **credentials, channels=fluorescence_channels, colorList=['FF0000', '00FF00']),
        OmeroRoISource(image_id, **credentials)
    )

    with VideoExporter('outpy.avi', framerate=3) as ve:
        print('Loading data from server...')
        for i, (image, overlay) in enumerate(tqdm.tqdm(irs)):
            mask = overlay.toMasks(*image.shape[:2])[0]

            masked_image = image * mask[:,:,None] 
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(osp.join('images', '%02d.png' % i), masked_image)
            ve.write(masked_image)