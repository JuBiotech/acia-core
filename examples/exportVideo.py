from acia.base import ImageSequenceSource, Overlay
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
import getpass
import cv2
import numpy as np
import tqdm
from acia.segm.omero.utils import ScaleBar

from acia.segm.omero.utils import ScaleBar
from acia.segm.output import VideoExporter

no_crop = lambda frame, overlay: (slice(0,frame.shape[0]), slice(0, frame.shape[1]))
def renderVideo(imageSource: ImageSequenceSource, roiSource=None, filename='output.mp4', framerate=3, codec="vp09", scaleBar: ScaleBar=None, draw_frame_number=False, cropper=no_crop):
    """Render a video of the time-lapse.

    Args:
        imageSource (ImageSequenceSource): Your time-lapse source object.
        roiSource ([type]): Your source of RoIs for the image (e.g. cells). If None, no RoIs are visualized. Defaults to None.
        filename (str, optional): The output path of the video. Defaults to 'output.mp4'.
        framerate (int, optional): The framerate of the video. E.g. 3 means three time-lapse images per second. Defaults to 3.
        codec (str, optional): The video format codec. Defaults to "vp09".
        scaleBar (ScaleBar, optional): The scale bar object. Defaults to None.
        draw_frame_number (bool, optional): Whether to draw the frame number. Defaults to False.
        cropper ([type], optional): The frame cropper object. Defaults to no_crop.
    """

    if roiSource is None:
        roiSource = [None] * len(imageSource)

    with VideoExporter(filename, framerate=framerate, codec=codec) as ve:
        for frame, (image, overlay) in enumerate(tqdm.tqdm(zip(imageSource, roiSource))):

            crop_parameters = cropper(image, overlay)
            image = image[crop_parameters[0], crop_parameters[1]]

            # draw overlay on image
            #im = Image.fromarray(image)
            #overlay.draw(im, (255, 255, 0))
            #image = np.asarray(image)
            height, width = image.shape[:2]

            # TODO: Draw float based contours
            if overlay:
                image = cv2.drawContours(image, [np.array(cont.coordinates).astype(np.int32) for cont in overlay.croppedContours(crop_parameters)], -1, (255, 255, 0)) # RGB format

            if draw_frame_number:
                cv2.putText(image, f'Frame: {frame}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            if scaleBar:
                image = scaleBar.draw(image, width - scaleBar.pixelWidth - 10, height - 10)

            # output images
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('images/%02d.png' % frame, image)
            ve.write(image)


class InitialCenterCrop:

    def __init__(self, width: int, height: int):
        self.width = width
        self.heigth = height

        self.image_center = None

    def __call__(self, image, overlay: Overlay):
        if self.image_center is None:
            self.image_center = np.round(np.mean([det.center for det in overlay], axis=0)).astype(np.int)

        half_height = int(np.round(self.heigth/2))
        half_width = int(np.round(self.width/2))

        x,y = self.image_center[0], self.image_center[1]
        return (slice(y - half_height, y + half_height), slice(x - half_width, x + half_width))

        


if __name__ == '__main__':
    serverUrl = 'ibt056'
    username = 'root'
    password = getpass.getpass(f'Password for {username}@{serverUrl}: ')
    framerate = 2

    imageId = 259 
    draw_frame_number = False
    draw_scale_bar = True
    cropping = InitialCenterCrop(125, 125)
    #cropping = lambda frame, overlay: (slice(0,frame.shape[0]), slice(0, frame.shape[1]))

    # load data from omero
    oss = OmeroSequenceSource(imageId, username=username, password=password, serverUrl=serverUrl, channels=[1], colorList=['FFFFFF'])
    #oss = OmeroSequenceSource(imageId, username=username, password=password, serverUrl=serverUrl, channels=[1], colorList=['FFFFFF'])
    ors = OmeroRoISource(imageId, username=username, password=password, serverUrl=serverUrl, range=(list(range(14))))

    assert len(oss) == len(ors)

    scaleBar = None
    if draw_scale_bar:
        scaleBar = ScaleBar(oss, 5, "MICROMETER", font_size=10)

    renderVideo(oss, ors, "outpy.mp4", codec="vp09", framerate=framerate, scaleBar=scaleBar, draw_frame_number=draw_frame_number, cropper=cropping)


