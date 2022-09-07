""" Deprecated example """

import omero
from omero.gateway import BlitzGateway

from acia.segm.omero.shapeUtils import create_ellipse, create_polygon, create_rectangle

username = "root"
password = "omero"


def print_user_info(conn):
    user = conn.getUser()
    print("Current user:")
    print("   ID:", user.getId())
    print("   Username:", user.getName())
    print("   Full Name:", user.getFullName())

    # Check if you are an Administrator
    print("   Is Admin:", conn.isAdmin())
    if not conn.isFullAdmin():
        # If 'Restricted Administrator' show privileges
        print(conn.getCurrentAdminPrivileges())

    print("Member of:")
    for g in conn.getGroupsMemberOf():
        print("   ID:", g.getName(), " Name:", g.getId())
    group = conn.getGroupFromContext()
    print("Current group: ", group.getName())

    # List the group owners and other members
    owners, members = group.groupSummary()
    print("   Group owners:")
    for o in owners:
        print(f"     ID: {o.getId()} {o.getOmeName()} Name: {o.getFullName()}")
        print("   Group members:")
    for m in members:
        print(f"     ID: {m.getId()} {m.getOmeName()} Name: {m.getFullName()}")

    print("Owner of:")
    for g in conn.listOwnedGroups():
        print("   ID: ", g.getName(), " Name:", g.getId())

    # Added in OMERO 5.0
    print("Admins:")
    for exp in conn.getAdministrators():
        print(f"   ID: {exp.getId} {exp.getOmeName()} Name: {exp.getFullName()}")

    # The 'context' of our current session
    ctx = conn.getEventContext()
    print(ctx)  # for more info


def print_obj(obj, indent=0):
    """
    Helper method to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    print(
        f"{' ' * indent}{obj.OMERO_CLASS}:{obj.getId()}  Name:'{obj.getName()}' (owner={obj.getOwnerOmeName()})"
    )


def list_projects(conn):
    """
    List projects with datasets and image sequences
    """
    # Load first 5 Projects, filtering by default group and owner
    my_exp_id = conn.getUser().getId()
    default_group_id = conn.getEventContext().groupId
    for project in conn.getObjects(
        "Project",
        opts={
            "owner": my_exp_id,
            "group": default_group_id,
            "order_by": "lower(obj.name)",
            "limit": 5,
            "offset": 0,
        },
    ):
        print_obj(project)
        # We can get Datasets with listChildren, since we have the Project already.
        # Or conn.getObjects("Dataset", opts={'project', id}) if we have Project ID
        for dataset in project.listChildren():
            print_obj(dataset, 2)
            for image in dataset.listChildren():
                print_obj(image, 4)


def inspect_image_by_id(conn, imageId: int):
    """
    Retrieve basic image information and render image to file
    """
    # Pixels and Channels will be loaded automatically as needed
    image = conn.getObject("Image", imageId)
    print(image.getName(), image.getDescription())
    # Retrieve information about an image.
    print(" X:", image.getSizeX())
    print(" Y:", image.getSizeY())
    print(" Z:", image.getSizeZ())
    print(" C:", image.getSizeC())
    print(" T:", image.getSizeT())
    # List Channels (loads the Rendering settings to get channel colors)
    for channel in image.getChannels():
        print("Channel:", channel.getLabel())
        print("Color:", channel.getColor().getRGB())
        print("Lookup table:", channel.getLut())
        print("Is reverse intensity?", channel.isReverseIntensity())

    # render the first timepoint, mid Z section
    z = 0  # image.getSizeZ() / 2
    t = 10
    rendered_image = image.renderImage(z, t)
    # rendered_image.show()               # popup (use for debug only)
    rendered_image.save("test.png")  # save in the current folder


# We have a helper function for creating an ROI and linking it to new shapes
def create_roi(updateService, img, shapes):
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return updateService.saveAndReturnObject(roi)


def add_roi_to_image(conn, imageId: int, z: int = 0, t: int = 0):
    updateService = conn.getUpdateService()
    image = conn.getObject("Image", imageId)

    shapes = [
        create_rectangle(0, 0, 100, 100, z=z, t=t),
        create_ellipse(150, 0, 50, 50, z=z, t=t),
        create_polygon([(10, 20), (50, 150), (200, 200), (250, 75)], z=z, t=t),
    ]

    create_roi(updateService, image, shapes)


def list_rois_for_image(conn, imageId):
    roi_service = conn.getRoiService()
    result = roi_service.findByImage(imageId, None)
    for roi in result.rois:
        print("ROI:  ID:", roi.getId().getValue())
        for s in roi.copyShapes():
            shape = {}
            shape["id"] = s.getId().getValue()
            shape["theT"] = s.getTheT().getValue()
            shape["theZ"] = s.getTheZ().getValue()
            if s.getTextValue():
                shape["textValue"] = s.getTextValue().getValue()
            if isinstance(s, omero.model.RectangleI):
                shape["type"] = "Rectangle"
                shape["x"] = s.getX().getValue()
                shape["y"] = s.getY().getValue()
                shape["width"] = s.getWidth().getValue()
                shape["height"] = s.getHeight().getValue()
            elif isinstance(s, omero.model.EllipseI):
                shape["type"] = "Ellipse"
                shape["x"] = s.getX().getValue()
                shape["y"] = s.getY().getValue()
                shape["radiusX"] = s.getRadiusX().getValue()
                shape["radiusY"] = s.getRadiusY().getValue()
            elif isinstance(s, omero.model.PointI):
                shape["type"] = "Point"
                shape["x"] = s.getX().getValue()
                shape["y"] = s.getY().getValue()
            elif isinstance(s, omero.model.LineI):
                shape["type"] = "Line"
                shape["x1"] = s.getX1().getValue()
                shape["x2"] = s.getX2().getValue()
                shape["y1"] = s.getY1().getValue()
                shape["y2"] = s.getY2().getValue()
            elif isinstance(s, omero.model.MaskI):
                shape["type"] = "Mask"
                shape["x"] = s.getX().getValue()
                shape["y"] = s.getY().getValue()
                shape["width"] = s.getWidth().getValue()
                shape["height"] = s.getHeight().getValue()
            elif isinstance(s, (omero.model.LabelI, omero.model.PolygonI)):
                print(type(s), " Not supported by this code")
            # Do some processing here, or just print:
            print(
                "   Shape:",
            )
            for key, value in shape.items():
                print(
                    "  ",
                    key,
                    value,
                )
            print("")


# context manager based Blitz connection to omero
with BlitzGateway(username, password, host="ibtomero", port=4064) as omero_conn:
    print_user_info(omero_conn)
    # list_projects(conn)
    # inspect_image_by_id(conn, 2)
    # add_roi_to_image(conn, 2)
    # list_rois_for_image(conn, 2)
