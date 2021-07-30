from distribution.omero.storer import OmeroRoIStorer

overlay = OmeroRoIStorer.load(2, 'root', 'omero', 'ibt056')

print("Num contours", len(overlay.contours))