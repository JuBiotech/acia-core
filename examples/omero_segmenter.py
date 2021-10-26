import getpass
import logging
from numpy import integer
from omero.gateway import BlitzGateway
from acia.base import Processor
from acia.segm.omero.storer import OmeroRoIStorer, OmeroSequenceSource
from acia.segm.omero.utils import image_iterator, list_images_in_dataset, list_images_in_project
from acia.segm.processor.offline import OfflineModel
from acia.segm.filter import NMSFilter
from examples.training_dataset.exportDataset import has_all_tags
import time
import omero
import itertools

def get_tags(conn, object):
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


def remove_tag(conn, object, tag_name: str):
    """ Removes the named tag from the omero object using the omero connection.

    Args:
        conn ([type]): omero connection
        object ([type]): omero object
        tag_name (str): tag name
    """
    tags = get_tags(conn, object)
    for tag in tags:
        if tag.getTextValue() == tag_name:
            # delete tag
            conn.deleteObjects('ImageAnnotationLink', [tag.link.getId()], wait=True)

            # work done
            return

    # war user that the tag could not be found
    logging.warn(f'tag "{tag_name}" not found in object "{object.getName()}"')

def create_model():
    # create local machine learning model
    return OfflineModel('model_zoo/htc/htc_tuned.py', 'model_zoo/htc/latest.pth', half=True, tiling={'x_shift': 768-128, 'y_shift': 768-128, 'tile_height': 768, 'tile_width': 768})

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
    print("Perform Prediction...")
    result = model.predict(source)

    # filter cell detections
    print("Filter detections")
    result = NMSFilter.filter(result, iou_thr=0.5, mode='i')

    # store detections in omero
    print("Save results...")
    OmeroRoIStorer.storeWithConn(result, imageId, conn=conn)

def segmentation(username, password, serverUrl):
    tag_filter = ['request-segm'] # e.g. add E. coli here

    request_tags = tag_filter

    model = None

    def get_model():
        return create_model()


    logging.info("Connect to omero...")
    with BlitzGateway(username, password, host=serverUrl, port=4064, secure=True) as conn:
        omero_objects = itertools.chain(conn.getObjects("Image"), conn.getObjects("Dataset"), conn.getObjects("Project"))

        # iterate over any kind of omero object (Image, Dataset, Project)
        for obj in omero_objects:
            if has_all_tags(obj, tag_filter):
                print(f'Found segmentation deletion request for {obj.getName()}')

                # iterate over the image(s) in there
                for image in image_iterator(conn, obj):
                    
                    # do the segmentation here
                    predict(image.getId(), get_model(), conn)
                    print("Segmentation successfull!")

                # if everything worked well: remove the tag that triggers the execution
                for tag in request_tags:
                    remove_tag(conn, obj, tag)


def del_segmentations(username, password, serverUrl):
    tag_filter = ['request-del-segm'] # e.g. add E. coli here

    request_tags = tag_filter

    with BlitzGateway(username, password, host=serverUrl, port=4064, secure=True) as conn:
        omero_objects = itertools.chain(conn.getObjects("Image"), conn.getObjects("Dataset"), conn.getObjects("Project"))

        # iterate over any kind of omero object (Image, Dataset, Project)
        for obj in omero_objects:
            if has_all_tags(obj, tag_filter):
                print(f'Found segmentation for {obj.getName()}')
                # iterate over the image(s) in there
                for image in image_iterator(conn, obj):
                    OmeroRoIStorer.clear(image.getId(), conn=conn)

                # if everything worked well: remove the tag that triggers the execution
                for tag in request_tags:
                    remove_tag(conn, obj, tag)


if __name__ == '__main__':
    serverUrl = 'ibt056'
    username = 'root'
    password = getpass.getpass(f'Password for {username}@{serverUrl}: ')

    omero_cred = {
        'username': username,
        'serverUrl': serverUrl,
        'password': password
    }


    while True:

        segmentation(**omero_cred)
        del_segmentations(**omero_cred)
        print('Sleep')
        time.sleep(30)