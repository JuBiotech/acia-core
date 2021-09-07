from acia.segm.output import VideoExporter
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
from acia.base import ImageRoISource
import cv2
import numpy as np
import os.path as osp
import tqdm
import numpy.ma as ma
from PIL import Image, ImageDraw
import pandas as pd


def exportVideo(irs: ImageRoISource):
    with VideoExporter('outpy.avi', framerate=3) as ve:
        print('Loading data from server...')
        for i, (image, overlay) in enumerate(tqdm.tqdm(irs)):
            mask = overlay.toMasks(*image.shape[:2])[0]

            masked_image = image * mask[:,:,None] 
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(osp.join('images', '%02d.png' % i), masked_image)
            ve.write(masked_image)

def exportFluorescence(irs: ImageRoISource):
    datapoints = []

    print('Loading data from server...')

    # apply fluorescence analysis in parallel to all images
    for frame, data in enumerate(irs.apply_parallel_star(analyze_fluorescence)):
        # extract and store result data
        for r,g, rgb, id in data:
            datapoints.append((frame, r, g, *rgb, id))

    # build pandas dataframe from datapoints
    df = pd.DataFrame(datapoints, columns=['frame', 'red', 'green', 'r', 'g', 'b', 'id'])

    # store to files
    df.to_csv('datapoints.csv')
    df.to_pickle('datapoints.pkl')

'''
    Take image and overlay and extract the fluorescence signal of the individual cell objects
'''
def analyze_fluorescence(image, overlay):
    datapoints = []

    # create image (for masks)
    height, width = image.shape[:2]
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    # iterate all rois in the overlay
    for roi in enumerate(overlay):
        # clear mask image
        draw.rectangle((0, 0, width, height), fill=(0,))

        # draw cell mask
        roi_mask = roi._toMask(img, draw=draw)

        # mask image
        masked_image = image * roi_mask[:,:,None]

        # create masked array
        masked_cell = ma.masked_array(masked_image, mask=np.repeat(~roi_mask[:,:,None], 3, axis=-1))

        # norm the channel values
        masked_cell = masked_cell.astype(np.float32) / 255

        # compute average fluorescence responses
        average_red = np.mean(masked_cell[:,:,0].compressed())
        average_green = np.mean(masked_cell[:,:,1].compressed())

        # store the extracted data
        datapoints.append((average_red, average_green, (masked_cell[:,:,0].compressed(), masked_cell[:,:,1].compressed(), masked_cell[:,:,2].compressed(), roi.id)))

    # return extracted data per point
    return datapoints


if __name__ == '__main__':
    # omero id of the image (Image ID)
    image_id = 470
    # fluorescence channels you want to monitor (usually 1 is the phase contrast)
    fluorescence_channels = [2, 3]

    # your user credentials for omero
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

    # run applications
    exportFluorescence(irs)
    #exportVideo(irs)
