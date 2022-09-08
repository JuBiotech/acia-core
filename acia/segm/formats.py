""" Functions for different segmentation formats"""


import json
from typing import Tuple

import networkx as nx
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


def parse_simple_tracking(file_content: str) -> Tuple[Overlay, nx.DiGraph]:
    """Parse simple tracking format from file content string

    Args:
        file_content (str): simple tracking format file content

    Returns:
        Tuple[Overlay, nx.DiGraph]: segmentation overlay and tracking graph
    """

    data = json.loads(file_content)

    segmentation_data = data["segmentation"]
    tracking_data = data["tracking"]

    # deal with the segmentation first
    all_detections = []

    # create contours
    for det in segmentation_data:
        all_detections.append(
            Contour(det["contour"], -1, det["frame"], det["id"], det["label"])
        )

    segmentation_overlay = Overlay(all_detections)

    # deal with the tracking
    tracking = nx.DiGraph()

    # create graph from id links
    for link in tracking_data:
        tracking.add_edge(link["sourceId"], link["targetId"])

    return segmentation_overlay, tracking


def gen_simple_tracking(overlay: Overlay, tracking_graph: nx.Graph) -> str:
    """Create a simple tracking format from overlay and tracking graph

    Args:
        overlay (Overlay): segmentation overlay
        tracking_graph (nx.Graph): tracking graph

    Returns:
        str: simple tracking format string
    """

    segmentation_data = []
    for cont in overlay:

        coordinates = cont.coordinates

        if isinstance(coordinates, np.ndarray):
            coordinates = coordinates.tolist()

        segmentation_data.append(
            dict(label=cont.label, contour=coordinates, id=cont.id, frame=cont.frame)
        )

    tracking_data = []
    for edge in tracking_graph.edges:
        tracking_data.append(dict(sourceId=edge[0], targetId=edge[1]))

    simpleTracking = dict(segmentation=segmentation_data, tracking=tracking_data)

    return json.dumps(simpleTracking)
