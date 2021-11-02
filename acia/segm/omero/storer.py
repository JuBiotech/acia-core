import logging
from acia.segm.omero.shapeUtils import make_coordinates
import omero
from omero.gateway import BlitzGateway
from acia.base import Contour, ImageSequenceSource, Overlay, RoISource
from .shapeUtils import create_polygon
import numpy as np
from itertools import product
import tqdm


# We have a helper function for creating an ROI and linking it to new shapes
def create_roi(updateService, img, shapes):
    '''
        Helper function to create the roi object
        updateService: omero update Service
        img: omero image object (not the id)
        shapes: list of omero.model shapes

        returns: RoI object
    '''
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return updateService.saveAndReturnObject(roi)

# We have a helper function for creating an ROI and linking it to new shapes
def create_roi_fast(updateService, img, shapes):
    '''
        Helper function to create the roi object (without waiting)
        updateService: omero update Service
        img: omero image object (not the id)
        shapes: list of omero.model shapes
    '''
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    updateService.saveObject(roi)

class OmeroRoIStorer:
    '''
        Stores and loads overlay results in the roi format (readable by ImageJ)
    '''

    @staticmethod
    def storeWithConn(overlay, imageId: int, conn, force=False, z=0):
        # retrieve omero objects
        updateService = conn.getUpdateService()
        image = conn.getObject("Image", imageId)

        userId = conn.getUser().getId()
        imageOwnerId = image.getOwner().getId()

        if not force and userId != imageOwnerId:
            raise ValueError("You try to write to non-owned data. Enable 'force' option if you are sure to do that.")

        #OmeroRoIStorer.clear(imageId=imageId, username=username, password=password, serverUrl=serverUrl, port=port, secure=secure)

        size_t = image.getSizeT()
        size_z = image.getSizeZ()

        if overlay.numFrames() == size_t * size_z:
            # this is a linearized overlay
            logging.info('Linearized overlay: Use t and z')
            shapes = [
                create_polygon(cont.coordinates, z=cont.frame % size_z, t=np.floor(cont.frame/size_z), description="Score: %.2f" % cont.score) for cont in overlay
            ]

        else:
            shapes = [
                create_polygon(cont.coordinates, z=z, t=cont.frame, description="Score: %.2f" % cont.score) for cont in overlay
            ]

        for shape in tqdm.tqdm(shapes):
            create_roi_fast(updateService, image, [shape])

        logging.info(f"Stored overlay with {len(overlay)} rois for image '{image.getName()}'")

    @staticmethod
    def store(overlay: Overlay, imageId: int, username: str, password: str, serverUrl: str, port=4064, secure=True, force=False, z=0):
        '''
            Stores overlay results in omero

            overlay: the overlay to store
            imageId: omero id of the image sequence
            username: omero username
            password: omero password
            serverUrl: omero web address
            port: omero port (default: 4064)
        '''

        with BlitzGateway(username, password, host=serverUrl, port=port, secure=secure) as conn:
            OmeroRoIStorer.storeWithConn(overlay, imageId, conn, force, z)

    @staticmethod
    def load(imageId: int, username: str, password: str, serverUrl: str, port=4064, secure=True, roiId=None) -> Overlay:
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
        with BlitzGateway(username, password, host=serverUrl, port=port, secure=secure) as conn:
            # get the roi service
            roi_service = conn.getRoiService()
            result = roi_service.findByImage(imageId, None)

            image = image = conn.getObject("Image", imageId)

            size_t = image.getSizeT()
            size_z = image.getSizeZ()


            # loop rois
            for roi in result.rois:
                if (not roiId is None) and roi.getId() != roiId:
                    # if the roiId is specified, check whether we have the right one
                    continue

                # loop shapes inside roi
                for s in roi.copyShapes():
                    if type(s) == omero.model.PolygonI:
                        # extract important information
                        t = s.getTheT().getValue()
                        points = make_coordinates(s.getPoints().getValue())
                        score = -1.

                        if size_z > 1:
                            t = t * size_z + s.getTheZ().getValue()

                        id = s.getId().getValue()

                        # add contour element to overlay
                        cont = Contour(points, score, t, id=id)
                        overlay.add_contour(cont)

        # return the overlay
        return overlay

    @staticmethod
    def clear(imageId: int, username: str = None, password: str = None, serverUrl: str = None, port=4064, secure=True, roiId=None, conn=None):
        # open connection to omero
        with BlitzConn(username=username, password=password, serverUrl=serverUrl, port=port, secure=secure, conn=conn).make_connection() as conn:
            # get the roi service
            roi_service = conn.getRoiService()
            result = roi_service.findByImage(imageId, None)

            print(f"Deleting {len(result.rois)} rois...")

            # delete all RoIs in the image
            conn.deleteObjects("Roi", [roi.getId().getValue() for roi in result.rois], deleteAnns=True, deleteChildren=True, wait=True)        

class IngoreWithWrapper:
    def __init__(self, object):
        self.object = object

    def __getattr__(self,attr):
        return self.object.__getattribute__(attr)

    def __enter__(self):
        return self.object
    def __exit__(self, type, value, traceback):
        pass

class BlitzConn(object):
    '''
        Encapsulates standard omero behavior
    '''
    def __init__(self, username, password, serverUrl, port=4064, secure=True, conn=None):
        self.username = username
        self.password = password
        self.serverUrl = serverUrl
        self.port = port
        self.secure = secure

        self.conn = conn

    def make_connection(self):
        if self.conn:
            # we already have an existing conn object
            return IngoreWithWrapper(self.conn)
        else:
            # return a new connection
            return BlitzGateway(self.username, self.password, host=self.serverUrl, port=self.port, secure=self.secure)

    def __enter__(self):
        self.make_connection()

    def __exit__(self, type, value, traceback):
        pass


class OmeroSequenceSource(ImageSequenceSource, BlitzConn):
    '''
        Uses omero server as a source for images
    '''
    def __init__(self, imageId: int, username: str = None, password: str = None, serverUrl: str = None, port=4064, channels=[1], z=0, imageQuality=1.0, secure=True, colorList=['FFFFFF', None, None], range=None, conn=None):
        '''
            imageId: id of the image sequence
            username: omero username
            password: omero password
            serverUrl: omero server url
            port: omero port
            channels: list of image channels to activate (e.g. include fluorescence channels)
            z: focus plane
            imageQuality: quality of the rendered images (1.0=no compression, 0.0=super compression)
            base_channel: id of the phase contrast channel (visualized over all rgb channels)
        '''

        BlitzConn.__init__(self, username=username, password=password, serverUrl=serverUrl, port=port, secure=secure, conn=conn)

        self.imageId = imageId
        self.channels = channels
        self.z = z
        self.imageQuality = imageQuality
        self.colorList = colorList
        self.range = range

        assert len(self.channels) <= len(self.colorList), f"you must specify a color for every channel! You have {len(self.channels)} channels ({self.channels}) but only {len(self.colorList)} color(s) ({self.colorList}). Please update your colorList!"

    def imageName(self) -> str:
        '''
            returns the name of the image
        '''
        with self.make_connection() as conn:
            return conn.getObject('Image', self.imageId).getName()

    def datasetName(self) -> str:
        '''
            returns the name of the dataset
        '''
        with self.make_connection() as conn:
            return conn.getObject('Image', self.imageId).getParent().getName()

    def projectName(self) -> str:
        '''
            returns the name of the associated project
        '''
        with self.make_connection() as conn:
            return conn.getObject('Image', self.imageId).getProject().getName()

    def __iter__(self):
        with self.make_connection() as conn:
            # get the specified image
            image = conn.getObject("Image", self.imageId)
            # set grayscale mode
            image.setGreyscaleRenderingModel()

            # size_c = image.getSizeC()
            size_t = image.getSizeT()
            size_z = image.getSizeZ()

            # iterate over time
            for t,z in product(range(size_t), range(size_z)):

                index = t * size_z + z

                if self.range and not index in self.range:
                    continue

                image.setColorRenderingModel()
                image.setActiveChannels(self.channels, colors=self.colorList)
                rendered_image = image.renderImage(z, t, compression=self.imageQuality)

                yield np.asarray(rendered_image, dtype=np.uint8)

    def __len__(self):
        with self.make_connection() as conn:
            image = conn.getObject('Image', self.imageId)
            if self.range:
                return min(image.getSizeT() * image.getSizeZ(), len(self.range))
            return int(image.getSizeT() * image.getSizeZ())


class OmeroRoISource(BlitzConn, RoISource):
    def __init__(self, imageId: int, username: str, password: str, serverUrl: str, port=4064, z=0, secure=True, roiSelector=lambda rois: [rois[0]], range=None):
        BlitzConn.__init__(self, username=username, password=password, serverUrl=serverUrl, port=port, secure=secure)

        self.imageId = imageId

        self.roiSelector = roiSelector
        self.range = range

    def __iter__(self):
        # create connection to omero
        with self.make_connection() as conn:
            roi_service = conn.getRoiService()

            # find all rois for the image
            result = roi_service.findByImage(self.imageId, None)
            rois = result.rois


            if len(rois) == 0:
                # no rois found
                raise ValueError(f"No rois found for image {self.imageId}")

            # select a subset of rois
            rois = self.roiSelector(rois)

            # compose an overlay from the rois
            overlay = OmeroRoIStorer.load(self.imageId, username=self.username, password=self.password,
                                    serverUrl=self.serverUrl, port=self.port, secure=self.secure)

            # return overlay iterator over time
            return overlay.timeIterator(frame_range=self.range)

    def __len__(self) -> int:
        with self.make_connection() as conn:
            image = conn.getObject('Image', self.imageId)
            if self.range:
                min(image.getSizeT() * image.getSizeZ(), len(self.range))
            return image.getSizeT() * image.getSizeZ()
