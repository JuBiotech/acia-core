from itertools import islice
from acia.base import Overlay
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
import getpass
import cv2
import numpy as np
import tqdm
from PIL import Image
from acia.segm.omero.utils import ScaleBar

def renderVideo(imageSource, roiSource, filename='output.avi', framerate=3):
    """[summary]

    Args:
        imageSource ([type]): [description]
        roiSource ([type]): [description]
        filename (str, optional): path/to/the/output/video/file. Defaults to 'output.avi'.
        framerate (int, optional): Frames per second in the video. Defaults to 3.
    """
    out = None

    for frame, (image, overlay) in enumerate(tqdm.tqdm(zip(imageSource, roiSource))):
        if out is None:
            # create video renderer
            frame_height, frame_width = image.shape[:2]
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), framerate, (frame_width,frame_height))

        # draw overlay on image
        #im = Image.fromarray(image)
        #overlay.draw(im, (255, 255, 0))
        #image = np.asarray(image)
        image = cv2.drawContours(image, [np.round(np.array(cont.coordinates)).astype(np.int32) for cont in overlay.contours if len(cont.coordinates) > 0], -1, (255, 255, 0)) # RGB format

        # output images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%02d.png' % frame, image)
        out.write(image)

    out.release()

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
    draw_contours = True
    draw_frame_number = False
    draw_scale_bar = True
    cropping = InitialCenterCrop(125, 125)
    #cropping = lambda frame, overlay: (slice(0,frame.shape[0]), slice(0, frame.shape[1]))

    # load data from omero
    oss = OmeroSequenceSource(imageId, username=username, password=password, serverUrl=serverUrl, channels=[1], colorList=['FFFFFF'])
    #oss = OmeroSequenceSource(imageId, username=username, password=password, serverUrl=serverUrl, channels=[1], colorList=['FFFFFF'])
    ors = OmeroRoISource(imageId, username=username, password=password, serverUrl=serverUrl)

    assert len(oss) == len(ors)

    out = None
    scaleBar = None
    if draw_scale_bar:
        scaleBar = ScaleBar(oss, 5, "MICROMETER", font_size=10)

    # join frame images and overlay
    for frame, (image, overlay) in islice(enumerate(tqdm.tqdm(zip(oss, ors))), 13):
        if out is None:
            # create video renderer
            frame_height, frame_width = image.shape[:2]
            out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), framerate, (frame_width,frame_height))

        # crop the image
        crop_parameters = cropping(image, overlay)
        image = image[crop_parameters[0], crop_parameters[1]]

        height, width = image.shape[:2]

        # draw overlay on image
        # TODO: Draw float based contours
        if draw_contours:
            image = cv2.drawContours(image, [np.array(cont.coordinates).astype(np.int32) for cont in overlay.croppedContours(crop_parameters)], -1, (255, 255, 0)) # RGB format

        if draw_frame_number:
            cv2.putText(image, f'Frame: {frame}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        if draw_scale_bar:
            image = scaleBar.draw(image, width - scaleBar.pixelWidth - 10, height - 10)

        # output images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%04d.png' % frame, image)
        out.write(image)

    out.release()


