""" Example to export a video rendering of an OMERO image sequence"""

import getpass

import numpy as np

from acia.base import Overlay
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
from acia.segm.omero.utils import ScaleBar
from acia.segm.output import renderVideo


class InitialCenterCrop:
    """Class to crop image around center position"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.heigth = height

        self.image_center = None

    def __call__(self, image, overlay: Overlay):
        if self.image_center is None:
            self.image_center = np.round(
                np.mean([det.center for det in overlay], axis=0)
            ).astype(np.int)

        half_height = int(np.round(self.heigth / 2))
        half_width = int(np.round(self.width / 2))

        x, y = self.image_center[0], self.image_center[1]
        return (
            slice(y - half_height, y + half_height),
            slice(x - half_width, x + half_width),
        )


if __name__ == "__main__":
    serverUrl = "ibt056"
    username = "root"
    password = getpass.getpass(f"Password for {username}@{serverUrl}: ")
    framerate = 2

    imageId = 259
    draw_frame_number = False
    draw_scale_bar = True
    cropping = InitialCenterCrop(125, 125)
    # cropping = lambda frame, overlay: (slice(0,frame.shape[0]), slice(0, frame.shape[1]))

    # load data from omero
    oss = OmeroSequenceSource(
        imageId,
        username=username,
        password=password,
        serverUrl=serverUrl,
        channels=[1],
        colorList=["FFFFFF"],
    )
    # oss = OmeroSequenceSource(imageId, username=username, password=password, serverUrl=serverUrl, channels=[1], colorList=['FFFFFF'])
    ors = OmeroRoISource(
        imageId,
        username=username,
        password=password,
        serverUrl=serverUrl,
        range=(list(range(14))),
    )

    assert len(oss) == len(ors)

    scaleBar = None
    if draw_scale_bar:
        scaleBar = ScaleBar(oss, 5, "MICROMETER", font_size=10)

    renderVideo(
        oss,
        ors,
        "outpy.mp4",
        codec="vp09",
        framerate=framerate,
        scaleBar=scaleBar,
        draw_frame_number=draw_frame_number,
        cropper=cropping,
    )
