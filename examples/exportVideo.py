from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
import getpass
import cv2
import numpy as np
import tqdm

if __name__ == '__main__':
    serverUrl = 'ibt056'
    username = 'root'
    password = getpass.getpass(f'Password for {username}@{serverUrl}: ')
    framerate = 3

    imageId = 351

    # load data from omero
    oss = OmeroSequenceSource(imageId, username=username, password=password, serverUrl=serverUrl)
    ors = OmeroRoISource(imageId, username=username, password=password, serverUrl=serverUrl)

    assert len(oss) == len(ors)

    out = None

    # join frame images and overlay
    for frame, (image, overlay) in enumerate(tqdm.tqdm(zip(oss, ors))):
        if out is None:
            # create video renderer
            frame_height, frame_width = image.shape[:2]
            out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), framerate, (frame_width,frame_height))

        # draw overlay on image
        image = cv2.drawContours(image, [np.array(cont.coordinates) for cont in overlay.contours], -1, (0, 255, 255)) # BGR format

        # output images
        cv2.imwrite('images/%02d.png' % frame, image)
        out.write(image)

    out.release()


