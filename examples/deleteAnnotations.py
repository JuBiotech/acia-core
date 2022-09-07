"""
    Delete segmentations of an omero image
"""

from acia.segm.omero.storer import OmeroRoIStorer

if __name__ == "__main__":
    # imageId: We want to delete the RoIs from this image
    imageId = 262

    OmeroRoIStorer.clear(imageId, "root", "omero", "ibt056")
