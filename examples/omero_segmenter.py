"""Automated segmenter based on OMERO tags"""

import getpass
import itertools
import logging
import time

import omero
from numpy import integer
from omero.gateway import BlitzGateway
from omero.rtypes import unwrap

from acia.base import Processor
from acia.segm.filter import NMSFilter
from acia.segm.omero.storer import OmeroRoIStorer, OmeroSequenceSource
from acia.segm.omero.utils import image_iterator
from acia.segm.processor.offline import OfflineModel
from examples.training_dataset.exportDataset import has_all_tags

print("Loading...")


def get_tags(_, object) -> list:
    """Obtain all tags associated with an omero object

    Args:
        conn ([type]): omero connection
        object ([type]): omero object

    Returns:
        [type]: list of tag objects
    """
    tags = []
    for ann in object.listAnnotations():
        if ann.OMERO_TYPE == omero.model.TagAnnotationI:
            tags.append(ann)

    return tags


def remove_tag(conn, object, tag_name: str, tag_owner_id=None):
    """Removes the named tag from the omero object using the omero connection.

    Args:
        conn ([type]): omero connection
        object ([type]): omero object
        tag_name (str): tag name

        Adapted from: https://github.com/ome/omero-web/blob/86640b8e7f0580066059b1dae26041cbace41585/omeroweb/webclient/controller/container.py#L819
    """
    toDelete = []

    dtype = object.OMERO_CLASS.lower()

    tags = get_tags(conn, object)
    for tag in tags:
        if tag.getTextValue() == tag_name:
            # TODO: get dtype from object
            for al in tag.getParentLinks(dtype, [object.getId()]):
                if (
                    al is not None
                    and al.canDelete()
                    and (
                        tag_owner_id is None
                        or unwrap(al.details.owner.id) == tag_owner_id
                    )
                ):
                    toDelete.append(al._obj)

    # Need to group objects by class then batch delete
    linksByType = {}
    for obj in toDelete:
        objType = obj.__class__.__name__.rstrip("I")
        if objType not in linksByType:
            linksByType[objType] = []
        linksByType[objType].append(obj.id.val)
    for linkType, ids in linksByType.items():
        conn.deleteObjects(linkType, ids, wait=True)
    # if len(notFound) > 0:
    #    raise AttributeError("Attribute not specified. Cannot be removed.")

    if len(toDelete) == 0:
        # warn user that the tag could not be found
        logging.warning('tag "%s" not found in object "%s"', tag_name, object.getName())


def create_model():
    # create local machine learning model
    return OfflineModel(
        "model_zoo/htc/version3/htc_like_def_detr_tuned.py",
        "model_zoo/htc/version3/latest.pth",
        half=True,
        tiling={
            "x_shift": 768 - 128,
            "y_shift": 768 - 128,
            "tile_height": 768,
            "tile_width": 768,
        },
    )


def predict(imageId: integer, model: Processor, conn):
    """[summary]

    Args:
        imageId (integer): omero id of the image
        model (Processor): model processor for segmentation
        conn ([type]): omero connection
    """
    # create local image data source
    source = OmeroSequenceSource(imageId, conn=conn)

    # perform overlay prediction
    print(f"Perform Prediction on {imageId}...")
    result = model.predict(source)

    # filter cell detections
    print("Filter detections")
    result = NMSFilter.filter(result, iou_thr=0.5, mode="i")

    # store detections in omero
    print("Save results...")
    OmeroRoIStorer.storeWithConn(result, imageId, conn=conn)


def segmentation(username, password, serverUrl):
    tag_filter = ["request-segm"]  # e.g. add E. coli here

    request_tags = tag_filter

    def get_model():
        return create_model()

    logging.info("Connect to omero...")
    with BlitzGateway(
        username, password, host=serverUrl, port=4064, secure=True
    ) as conn:
        # TODO: get really all objects!

        # Querying across omero groups
        conn.SERVICE_OPTS.setOmeroGroup("-1")

        omero_objects = itertools.chain(
            conn.getObjects("Project"),
            conn.getObjects("Image"),
            conn.getObjects("Dataset"),
        )  # *all_objects)

        # iterate over any kind of omero object (Image, Dataset, Project)
        for obj in omero_objects:
            if has_all_tags(obj, tag_filter):
                print(f"Found segmentation request for {obj.getName()}")
                request_segm_tag = list(
                    filter(
                        lambda tag: tag.OMERO_TYPE == omero.model.TagAnnotationI
                        and tag.getTextValue() == "request-segm",
                        get_tags(conn, obj),
                    )
                )
                assert len(request_segm_tag) == 1
                request_segm_tag = request_segm_tag[0]
                owner = request_segm_tag.link.getDetails().getOwner().getName()

                # make a user connection the lives for 10 minutes
                # TODO: make a user connection the lives for 24 hours (seems to be strangely counted...)
                user_connection = conn.suConn(owner, ttl=24 * 60 * 60000)

                # iterate over the image(s) in there
                for image in image_iterator(conn, obj):

                    # TODO: define shapeType
                    roi_count = image.getROICount()

                    if roi_count > 0:
                        # TODO: add a force/overwrite option
                        logging.info(
                            "Skip %s because it already has %d RoIs detected.",
                            image.getName(),
                            roi_count,
                        )
                        continue

                    # do the segmentation here
                    predict(image.getId(), get_model(), user_connection)
                    print("Segmentation successfull!")

                # if everything worked well: remove the tag that triggers the execution
                for tag in request_tags:
                    remove_tag(user_connection, obj, tag)

                user_connection.close()


def del_segmentations(username, password, serverUrl):
    tag_filter = ["request-del-segm"]  # e.g. add E. coli here

    request_tags = tag_filter

    with BlitzGateway(
        username, password, host=serverUrl, port=4064, secure=True
    ) as conn:

        # Querying across omero groups
        conn.SERVICE_OPTS.setOmeroGroup("-1")

        omero_objects = itertools.chain(
            conn.getObjects("Image"),
            conn.getObjects("Dataset"),
            conn.getObjects("Project"),
        )

        # iterate over any kind of omero object (Image, Dataset, Project)
        for obj in omero_objects:
            if has_all_tags(obj, tag_filter):
                print(f"Found segmentation deletion request for {obj.getName()}")

                request_del_segm_tag = list(
                    filter(
                        lambda tag: tag.OMERO_TYPE == omero.model.TagAnnotationI
                        and tag.getTextValue() == "request-del-segm",
                        get_tags(conn, obj),
                    )
                )
                assert len(request_del_segm_tag) == 1
                request_del_segm_tag = request_del_segm_tag[0]
                owner = request_del_segm_tag.link.getDetails().getOwner().getName()

                # make a user connection the lives for 10 minutes
                # TODO: same strange user connection init
                user_connection = conn.suConn(owner, ttl=24 * 60 * 60000)

                # iterate over the image(s) in there
                for image in image_iterator(conn, obj):
                    OmeroRoIStorer.clear(image.getId(), conn=user_connection)

                # if everything worked well: remove the tag that triggers the execution
                for tag in request_tags:
                    remove_tag(user_connection, obj, tag)

                user_connection.close()


if __name__ == "__main__":
    serverUrl = "ibt056"
    username = "root"
    password = getpass.getpass(f"Password for {username}@{serverUrl}: ")

    omero_cred = {"username": username, "serverUrl": serverUrl, "password": password}

    while True:

        segmentation(**omero_cred)
        del_segmentations(**omero_cred)
        print("Sleep")
        time.sleep(30)
