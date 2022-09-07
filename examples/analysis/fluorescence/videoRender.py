""" Example to render fluorescence video """

import os.path as osp

import cv2
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from config import basepath
from PIL import Image

from acia.base import ImageRoISource
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
from acia.segm.output import VideoExporter


def main():

    image_id = 470

    # your user credentials
    credentials = dict(
        username="root",
        password="omero",
        serverUrl="ibt056",
    )

    # combine images and rois
    irs = ImageRoISource(
        OmeroSequenceSource(
            image_id,
            **credentials,
            channels=[1, 2, 3],
            colorList=["FFFFFF", "FF0000", "00FF00"]
        ),
        OmeroRoISource(image_id, **credentials),
    )

    # read dataset
    df = pd.read_pickle(osp.join(basepath, "datapoints.pkl"))
    cell_index = 0

    with VideoExporter(osp.join(basepath, "cell_clustering.avi"), framerate=3) as ve:
        print("Loading data from server...")
        for image, overlay in tqdm.tqdm(irs):
            # draw all cell countours with their respective cluster color
            pil_image = Image.fromarray(image, "RGB")
            for roi in overlay:
                roi.draw(pil_image, outlineColor=df["color"][cell_index])
                cell_index += 1

            # convert to raw image
            raw_image = np.asarray(pil_image)

            # convert to bgr (for opencv output)
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

            # add frame to video
            ve.write(raw_image)


if __name__ == "__main__":
    main()
