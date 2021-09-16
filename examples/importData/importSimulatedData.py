'''
    This file shows how to add RoIs to existing omero images.

    The specific example is that we have simulated time-lapse images
    and the image data is already present in OMERO but the RoIs are
    still in the local format. Images are augmented with RoIs from the local source.
'''

from acia.segm.omero.storer import OmeroRoIStorer
from acia.segm.omero.utils import list_image_ids_in_dataset, list_images_in_dataset
from omero.gateway import BlitzGateway
from acia.segm.local import ImageJRoISource, LocalSequenceSource, RoiStorer
from acia.base import ImageRoISource, ImageSequenceSource
import os
import glob

if __name__ == '__main__':

    # path of simulated data images
    simulated_data_path = 'simulated_data'

    # dataset id (where the simulated image stacks reside)
    dataset_id = 451

    # credentials
    credentials = dict(
        username='root',
        host='ibt056',
        port=4064,
        secure=True,
        passwd='omero'
    )

    credentials2 = dict(
        serverUrl='ibt056',
        username='root',
        password='omero'
    )

    print("Connect to omero...")
    with BlitzGateway(**credentials) as conn:
        # list all images in the dataset
        image_list = list_images_in_dataset(conn, dataset_id)
        # create a lookup name -> id
        image_lookup ={image.getName(): image.getId() for image in image_list}

    # iterate all local files
    for filename in glob.glob(os.path.join(simulated_data_path, '*.tif')):
        # create roi source
        ijrs = ImageJRoISource(filename)
        # lookup omero image id
        image_id = image_lookup[os.path.basename(filename)]

        # upload RoIs
        print(f"Filename: {filename}, imageId: {image_id}, overlay size: {len(ijrs.overlay.contours)}")
        OmeroRoIStorer.store(ijrs.overlay, image_id, **credentials2)