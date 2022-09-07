"""Classes for OMERO storage interaction"""

import logging
from typing import Tuple

import numpy as np
import omero
import tqdm.auto as tqdm
from omero.gateway import BlitzGateway
from omero.model import LengthI

from acia.base import BaseImage, Contour, ImageSequenceSource, Overlay, RoISource
from acia.segm.local import LocalImage
from acia.segm.omero.shapeUtils import make_coordinates
from acia.segm.utils import compute_indices

from .shapeUtils import create_polygon


# We have a helper function for creating an ROI and linking it to new shapes
def create_roi(updateService, img, shapes):
    """
    Helper function to create the roi object
    updateService: omero update Service
    img: omero image object (not the id)
    shapes: list of omero.model shapes

    returns: RoI object
    """
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
    """
    Helper function to create the roi object (without waiting)
    updateService: omero update Service
    img: omero image object (not the id)
    shapes: list of omero.model shapes
    """
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    updateService.saveObject(roi)


class OmeroRoIStorer:
    """
    Stores and loads overlay results in the roi format (readable by ImageJ)
    """

    @staticmethod
    def storeWithConn(overlay, imageId: int, conn, force=False):
        # retrieve omero objects
        updateService = conn.getUpdateService()
        image = conn.getObject("Image", imageId)

        userId = conn.getUser().getId()
        imageOwnerId = image.getOwner().getId()

        if not force and userId != imageOwnerId:
            raise ValueError(
                "You try to write to non-owned data. Enable 'force' option if you are sure to do that."
            )

        # OmeroRoIStorer.clear(imageId=imageId, username=username, password=password, serverUrl=serverUrl, port=port, secure=secure)

        # size_t = image.getSizeT()
        size_z = image.getSizeZ()

        # this is a linearized overlay
        logging.info(
            "Using Linearized overlay: [t_1: z_0, z_1, ... z_Z, t_2: ,...] Use t and z"
        )
        shapes = [
            create_polygon(
                cont.coordinates,
                z=cont.frame % size_z,
                t=np.floor(cont.frame / size_z),
                description=f"Score: {cont.score:.2f}",
            )
            for cont in overlay
        ]

        for shape in tqdm.tqdm(shapes):
            create_roi_fast(updateService, image, [shape])

        logging.info(
            "Stored overlay with %d rois for image '%s'", len(overlay), image.getName()
        )

    @staticmethod
    def store(
        overlay: Overlay,
        imageId: int,
        username: str,
        password: str,
        serverUrl: str,
        port=4064,
        secure=True,
        force=False,
        conn=None,
    ):
        """
        Stores overlay results in OMERO. Uses existing connection if available.

        overlay: the overlay to store
        imageId: omero id of the image sequence
        username: omero username
        password: omero password
        serverUrl: omero web address
        port: omero port (default: 4064)
        conn: existing OMERO connection or None. (default: None)
        """

        with BlitzConn(username, password, serverUrl, port, secure, conn) as omero_conn:
            OmeroRoIStorer.storeWithConn(overlay, imageId, omero_conn, force)

    @staticmethod
    def load(
        imageId: int,
        username: str,
        password: str,
        serverUrl: str,
        port=4064,
        secure=True,
        roiId=None,
    ) -> Overlay:
        """
        Loads overlay from omero. Only considers polygons.

        imageId: omero id of the image sequence
        username: omero username
        password: omero password
        serverUrl: omero web address
        port: omero port (default: 4064)
        """
        overlay = Overlay([])
        # open connection to omero
        with BlitzGateway(
            username, password, host=serverUrl, port=port, secure=secure
        ) as conn:
            # get the roi service
            roi_service = conn.getRoiService()
            result = roi_service.findByImage(imageId, None)

            image = image = conn.getObject("Image", imageId)

            # size_t = image.getSizeT()
            size_z = image.getSizeZ()

            # loop rois
            for roi in result.rois:
                if (roiId is not None) and roi.getId() != roiId:
                    # if the roiId is specified, check whether we have the right one
                    continue

                # loop shapes inside roi
                for s in roi.copyShapes():
                    if isinstance(s, omero.model.PolygonI):
                        # extract important information
                        t = s.getTheT().getValue()
                        points = make_coordinates(s.getPoints().getValue())
                        score = -1.0

                        if size_z > 1:
                            t = t * size_z + s.getTheZ().getValue()

                        id = s.getId().getValue()

                        label = s.getTextValue().getValue()

                        # add contour element to overlay
                        cont = Contour(points, score, t, id=id, label=label)
                        overlay.add_contour(cont)

        # return the overlay
        return overlay

    @staticmethod
    def clear(
        imageId: int,
        username: str = None,
        password: str = None,
        serverUrl: str = None,
        port=4064,
        secure=True,
        conn=None,
    ):
        # open connection to omero
        with BlitzConn(
            username=username,
            password=password,
            serverUrl=serverUrl,
            port=port,
            secure=secure,
            conn=conn,
        ).make_connection() as omero_conn:
            # get the roi service
            roi_service = omero_conn.getRoiService()
            updateService = omero_conn.getUpdateService()
            result = roi_service.findByImage(imageId, None)

            print(f"Deleting {len(result.rois)} rois...")

            for roi in result.rois:
                shapes = roi.copyShapes()
                if len(shapes) > 1:
                    for s in roi.copyShapes():
                        roi.removeShape(s)
                    roi = updateService.saveAndReturnObject(roi)

            # delete all RoIs in the image
            if len(result.rois) > 0:
                omero_conn.deleteObjects(
                    "Roi",
                    [roi.getId().getValue() for roi in result.rois],
                    deleteAnns=True,
                    deleteChildren=True,
                    wait=True,
                )


class IngoreWithWrapper:
    """Wrapper to ignore context and do not exit after leaving. Is used for not starting/ending omero connections all the time!"""

    def __init__(self, object):
        self.object = object

    def __getattr__(self, attr):
        return self.object.__getattribute__(attr)

    def __enter__(self):
        return self.object

    def __exit__(self, type, value, traceback):
        pass


class BlitzConn:
    """
    Encapsulates standard omero behavior
    """

    def __init__(
        self, username, password, serverUrl, port=4064, secure=True, conn=None
    ):

        assert username is not None, "Please provide a username"
        assert password is not None, "Please provide a password"
        assert serverUrl is not None, "Please provide a OMERO server"

        self.username = username
        self.password = password
        self.serverUrl = serverUrl
        self.port = port
        self.secure = secure

        self.conn = conn

    def make_connection(self):
        # try to keep connection alive
        if self.conn is not None and self.conn.keepAlive():
            return IngoreWithWrapper(self.conn)
        else:
            # return a new connection
            conn = BlitzGateway(
                self.username,
                self.password,
                host=self.serverUrl,
                port=self.port,
                secure=self.secure,
            )
            conn.connect()
            conn.SERVICE_OPTS.setOmeroGroup("-1")
            self.conn = conn
            return IngoreWithWrapper(self.conn)

    def __enter__(self):
        self.make_connection()

    def __exit__(self, type, value, traceback):
        pass

    def __del__(self):
        # make sure the connection is always closed
        if self.conn:
            self.conn.__exit__()
            self.conn = None


class OmeroSource(BlitzConn):
    """
    Base Class for omero image information. Bundles functionality for image and RoIs.
    """

    def __init__(
        self,
        imageId: float,
        username: str = None,
        password: str = None,
        serverUrl: str = None,
        port=4064,
        secure=True,
        conn=None,
    ):
        """
        Args:
            imageId (float): omero image id
            username (str, optional): omero username. Not needed when conn is provided. Defaults to None.
            password (str, optional): omero password. Not needed when conn is provided. Defaults to None.
            serverUrl (str, optional): omero url. Not needed when conn is provided. Defaults to None.
            port (int, optional): omero port. Not needed when conn is provided. Defaults to 4064.
            secure (bool, optional): Whether to choose secure connection. Defaults to True.
            conn ([type], optional): Existing omero connection. Defaults to None.
        """
        BlitzConn.__init__(
            self,
            username=username,
            password=password,
            serverUrl=serverUrl,
            port=port,
            secure=secure,
            conn=conn,
        )

        self.imageId = imageId

    @property
    def rawPixelSize(self) -> Tuple[LengthI, LengthI]:
        """Return the pixel size in omero objects

        Returns:
            Tuple[LengthI,LengthI]: x and y pixel size in omero objects
        """
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)

            size_x_obj = image.getPixelSizeX(units="MICROMETER")
            size_y_obj = image.getPixelSizeY(units="MICROMETER")

            return size_x_obj, size_y_obj

    @property
    def pixelSize(self) -> Tuple[float, float]:
        """Return the pixel size in micron

        Returns:
            Tuple[float,float]: x and y pixel size in micron
        """
        size_x_obj, size_y_obj = self.rawPixelSize

        return size_x_obj.getValue(), size_y_obj.getValue()

    def printPixelSize(self, unit="MICROMETER"):
        """Output pixel sizes

        Args:
            unit (str, optional): Name of the unit. Defaults to "MICROMETER".
        """
        # get raw
        size_x_obj, size_y_obj = self.rawPixelSize

        # convert to correct unit
        size_x_obj = omero.model.LengthI(size_x_obj, unit)
        size_y_obj = omero.model.LengthI(size_y_obj, unit)

        # output pixel sizes
        print(f" Pixel Size X: {size_x_obj.getValue()} ({size_x_obj.getSymbol()})")
        print(f" Pixel Size Y: {size_y_obj.getValue()} ({size_y_obj.getSymbol()})")


class OmeroSequenceSource(ImageSequenceSource, OmeroSource):
    """
    Uses omero server as a source for images
    """

    def __init__(
        self,
        imageId: int,
        username: str = None,
        password: str = None,
        serverUrl: str = None,
        port=4064,
        channels=None,
        z=0,
        imageQuality=1.0,
        secure=True,
        colorList=None,
        range=None,
        conn=None,
    ):
        """
        imageId: id of the image sequence
        username: omero username
        password: omero password
        serverUrl: omero server url
        port: omero port
        channels: list of image channels to activate (e.g. include fluorescence channels)
        z: focus plane
        imageQuality: quality of the rendered images (1.0=no compression, 0.0=super compression)
        base_channel: id of the phase contrast channel (visualized over all rgb channels)
        """

        OmeroSource.__init__(
            self,
            imageId=imageId,
            username=username,
            password=password,
            serverUrl=serverUrl,
            port=port,
            secure=secure,
            conn=conn,
        )

        if channels is None:
            channels = [1]
        if colorList is None:
            colorList = ["FFFFFF", None, None]

        self.imageId = imageId
        self.channels = channels
        self.z = z
        self.imageQuality = imageQuality
        self.colorList = colorList
        self.range = range

        self.omero_image = None

        assert len(self.channels) <= len(
            self.colorList
        ), f"you must specify a color for every channel! You have {len(self.channels)} channels ({self.channels}) but only {len(self.colorList)} color(s) ({self.colorList}). Please update your colorList!"

    def imageName(self) -> str:
        """
        returns the name of the image
        """
        with self.make_connection() as conn:
            return conn.getObject("Image", self.imageId).getName()

    def datasetName(self) -> str:
        """
        returns the name of the dataset
        """
        with self.make_connection() as conn:
            return conn.getObject("Image", self.imageId).getParent().getName()

    def projectName(self) -> str:
        """
        returns the name of the associated project
        """
        with self.make_connection() as conn:
            return conn.getObject("Image", self.imageId).getProject().getName()

    def __get_omero_image(self):
        # open the connection
        self.conn = self.make_connection().__enter__()

        # cache the image
        if self.omero_image is None:
            self.omero_image = self.conn.getObject("Image", self.imageId)

        return self.omero_image

    def __get_image(self, frame: int) -> BaseImage:
        # get the specified image
        image = self.__get_omero_image()

        size_t = image.getSizeT()
        size_z = image.getSizeZ()

        t, z = compute_indices(frame, size_t, size_z)

        # perform rendering
        image.setColorRenderingModel()
        image.setActiveChannels(self.channels, colors=self.colorList)
        rendered_image = image.renderImage(z, t, compression=self.imageQuality)

        # return local image
        return LocalImage(np.asarray(rendered_image, dtype=np.uint8))

    def __iter__(self):
        for frame in range(self.num_frames):
            if self.range and frame not in self.range:
                continue
            yield self.get_frame(frame)

    def get_frame(self, frame: int):
        return self.__get_image(frame)

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def num_frames(self) -> int:
        return len(self)

    def __len__(self):
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            if self.range:
                return min(image.getSizeT() * image.getSizeZ(), len(self.range))
            return int(image.getSizeT() * image.getSizeZ())


class OmeroRoISource(OmeroSource, RoISource):
    """Source for OMERO RoIs beloging to an OMERO image sequence"""

    def __init__(
        self,
        imageId: int,
        username: str,
        password: str,
        serverUrl: str,
        port=4064,
        secure=True,
        roiSelector=lambda rois: [rois[0]],
        range=None,
        scale=None,
        conn=None,
    ):
        OmeroSource.__init__(
            self,
            imageId=imageId,
            username=username,
            password=password,
            serverUrl=serverUrl,
            port=port,
            secure=secure,
            conn=conn,
        )

        self.imageId = imageId

        self.roiSelector = roiSelector
        self.range = range
        self.scale = scale
        if self.scale:
            # 1 pixel has the size of the returned value. To move to correct domain use that size as scale factor
            self.scaleFactor = omero.model.LengthI(
                self.rawPixelSize[0], self.scale
            ).getValue()

        self.overlay = None

    def __iter__(self):
        # return overlay iterator over time
        return self.get_overlay().timeIterator(frame_range=self.range)

    def get_overlay(self):
        if self.overlay is None:
            # compose an overlay from the rois
            self.overlay = OmeroRoIStorer.load(
                self.imageId,
                username=self.username,
                password=self.password,
                serverUrl=self.serverUrl,
                port=self.port,
                secure=self.secure,
            )

            if self.scale:
                self.overlay.scale(self.scaleFactor)

        return self.overlay

    def __len__(self) -> int:
        with self.make_connection() as conn:
            image = conn.getObject("Image", self.imageId)
            if self.range:
                return min(image.getSizeT() * image.getSizeZ(), len(self.range))
            return image.getSizeT() * image.getSizeZ()
