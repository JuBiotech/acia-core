"""Example to crop image sequences"""

import os.path as osp

import cv2
import numpy as np

from acia.segm.local import ImageJRoISource, LocalSequenceSource

if __name__ == "__main__":

    local_tiff_file = osp.join("simulated_data/00.tif")

    rs = ImageJRoISource(local_tiff_file)
    ss = LocalSequenceSource(local_tiff_file)

    vertical_slice = slice(350, 550)
    horizontal_slice = slice(225, 425)

    for frame, (overlay, image) in enumerate(zip(rs, ss)):
        image = cv2.drawContours(
            image,
            [np.array(cont.coordinates).astype(np.int32) for cont in overlay.contours],
            -1,
            (255, 255, 0),
        )  # RGB format

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image[vertical_slice, horizontal_slice]

        cv2.imwrite(f"images/{frame:02d}.png", image)
