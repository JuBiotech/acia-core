""" Functions for different segmentation formats"""


import json

import numpy as np

from acia.base import Contour, Overlay


def parse_simple_segmentation(file_content: str) -> Overlay:
    """Parse simple segmentation format from string (json)

    Args:
        file_content (str): simple segmentation file content as string

    Returns:
        Overlay: the overlay representation of the segmentation
    """

    file_data = json.loads(file_content)

    contours = []

    for frame in file_data:
        frame_id = frame["frame"]
        for det in frame["detections"]:
            contours.append(
                Contour(det["contour"], -1.0, frame_id, det["id"], det["label"])
            )

    return Overlay(contours)


def gen_simple_segmentation(overlay: Overlay) -> str:
    """Create a simple segmentation string from an Overlay

    Args:
        overlay (Overlay): the overlay to store

    Returns:
        str: string containing the stringified simple segmentation json format
    """

    frame_packages = []

    # loop over frames
    for frame_overlay in overlay.timeIterator():
        if len(frame_overlay) == 0:
            continue

        det_objects = []
        frame_id = -1

        # loop over all contours in a frame
        for cont in frame_overlay:

            # transform coordinates to list (otherwise not json serializable)
            coordinates = cont.coordinates
            if isinstance(coordinates, np.ndarray):
                coordinates = coordinates.tolist()

            # create detection object
            det_objects.append(
                dict(
                    label=cont.label,
                    contour=coordinates,
                    id=cont.id,
                )
            )

            frame_id = cont.frame

        # create the frame package
        frame_package = dict(
            frame=frame_id,
            detections=det_objects,
        )
        frame_packages.append(frame_package)

    # serialize into json format
    return json.dumps(frame_packages, separators=(",", ":"))
