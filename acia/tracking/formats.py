""" Module to convert tracking formats """

import json
from typing import Tuple

import networkx as nx
import numpy as np

from acia.base import Contour, Overlay


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

    tracking.add_nodes_from(map(lambda ov: ov.id, segmentation_overlay))

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
