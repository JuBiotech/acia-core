from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
import getpass
import cv2
import numpy as np
import tqdm

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
        image = cv2.drawContours(image, [np.array(cont.coordinates) for cont in overlay.contours], -1, (255, 255, 0)) # RGB format

        # output images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%02d.png' % frame, image)
        out.write(image)

    out.release()


if __name__ == '__main__':
    serverUrl = 'ibt056'
    username = 'root'
    password = getpass.getpass(f'Password for {username}@{serverUrl}: ')
    framerate = 3

    draw_contours = True

    # load data from omero
    oss = OmeroSequenceSource(imageId, username=username, password=password, serverUrl=serverUrl, channels=[1, 2, 3], colorList=['FFFFFF', '770000', '007700'])
    ors = OmeroRoISource(imageId, username=username, password=password, serverUrl=serverUrl)

    assert len(oss) == len(ors)

    out = None

    scaleBar = ScaleBar(oss, 10, "MICROMETER")

    # join frame images and overlay
    for frame, (image, overlay) in enumerate(tqdm.tqdm(zip(oss, ors))):
        if out is None:
            # create video renderer
            frame_height, frame_width = image.shape[:2]
            out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), framerate, (frame_width,frame_height))

        height, width = image.shape[:2]

        # draw overlay on image
        # TODO: Draw float based contours
        if draw_contours:
            image = cv2.drawContours(image, [np.array(cont.coordinates).astype(np.int32) for cont in overlay.contours], -1, (255, 255, 0)) # RGB format

        cv2.putText(image, f'Frame: {frame}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        # TODO: Draw a scalebar
        image = scaleBar.draw(image, width - scaleBar.pixelWidth - 20, height - 20)

        # output images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('images/%02d.png' % frame, image)
        out.write(image)

    out.release()


