from acia.segm.omero.shapeUtils import make_coordinates
import omero
from omero.gateway import BlitzGateway
from acia.base import Contour, ImageSequenceSource, Overlay
from .shapeUtils import create_polygon
import numpy as np

from io import BytesIO
from PIL import Image

# We have a helper function for creating an ROI and linking it to new shapes
def create_roi(updateService, img, shapes):
    '''
        Helper function to create the roi object
        updateService: omero update Service
        img: omero image object (not the id)
        shapes: list of omero.model shapes
    '''
    import omero
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return updateService.saveAndReturnObject(roi)

class OmeroRoIStorer:
    '''
        Stores and loads overlay results in the roi format (readable by ImageJ)
    '''
    
    @staticmethod
    def store(overlay: Overlay, imageId: int, username: str, password: str, serverUrl: str, port=4064):
        '''
            Stores overlay results in omero

            overlay: the overlay to store
            imageId: omero id of the image sequence
            username: omero username
            password: omero password
            serverUrl: omero web address
            port: omero port (default: 4064)
        '''

        with BlitzGateway(username, password, host=serverUrl, port=port) as conn:
            # retrieve omero objects
            updateService = conn.getUpdateService()
            image = conn.getObject("Image", imageId)

            shapes = [
                create_polygon(cont.coordinates, z=0, t=cont.frame, description="Score: %.2f" % cont.score) for cont in overlay
            ]

            create_roi(updateService, image, shapes)

    @staticmethod
    def load(imageId: int, username: str, password: str, serverUrl: str, port=4064) -> Overlay:
        '''
            Loads overlay from omero. Only considers polygons.

            imageId: omero id of the image sequence
            username: omero username
            password: omero password
            serverUrl: omero web address
            port: omero port (default: 4064)
        '''
        overlay = Overlay()
        # open connection to omero
        with BlitzGateway(username, password, host=serverUrl, port=port) as conn:
            # get the roi service
            roi_service = conn.getRoiService()
            result = roi_service.findByImage(imageId, None)

            # loop rois
            for roi in result.rois:
                # loop shapes inside roi
                for s in roi.copyShapes():
                    if type(s) == omero.model.PolygonI:
                        # extract important information
                        t = s.getTheT().getValue()
                        points = make_coordinates(s.getPoints().getValue())
                        score = -1.

                        # add contour element to overlay
                        cont = Contour(points, score, t)
                        overlay.add_contour(cont)

        # return the overlay
        return overlay

class OmeroSequenceSource(ImageSequenceSource):
    '''
        Uses omero server as a source for images
    '''
    def __init__(self, imageId: int, username: str, password: str, serverUrl: str, port=4064, channels=[1], z=0, imageQuality=1.0):
        '''
            imageId: id of the image sequence
            username: omero username
            password: omero password
            serverUrl: omero server url
            port: omero port
            channels: list of image channels to activate (e.g. include fluorescence channels)
            z: focus plane
            imageQuality: quality of the rendered images (1.0=no compression, 0.0=super compression)
        '''

        self.imageId = imageId
        self.channels = channels
        self.z = z
        self.imageQuality = imageQuality

        # omero
        self.username = username
        self.password = password
        self.serverUrl = serverUrl
        self.port = port

    def __iter__(self):
        with BlitzGateway(self.username, self.password, host=self.serverUrl, port=self.port) as conn:
            # get the specified image
            image = conn.getObject("Image", self.imageId)
            # set grayscale mode
            image.setGreyscaleRenderingModel()

            size_c = image.getSizeC()
            size_t = image.getSizeT()
            z = self.z

            # iterate over time
            for t in range(size_t):
                # set active channels
                image.setActiveChannels(self.channels)
                # render the image
                rendered_image = image.renderImage(z, t, compression=self.imageQuality)
                # convert to rgb
                rendered_image = rendered_image.convert("RGB")
                # convert to numpy
                numpy_image = np.asarray(rendered_image, dtype=np.uint8)
                #rendered_image.save('test.jpg')

                # eject numpy image
                yield numpy_image