"""Example to load omero RoIs for an image sequence"""


from acia.segm.omero.storer import OmeroRoIStorer

overlay = OmeroRoIStorer.load(2, "root", "omero", "ibt056")

print("Num contours", len(overlay.contours))
