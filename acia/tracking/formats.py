""" Module to convert tracking formats """

import json
import re
from pathlib import Path
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
    for cont in segmentation_overlay:
        tracking.nodes[cont.id]["frame"] = cont.frame

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


def read_ctc_tracking(file: Path) -> list[dict]:
    """Reads a tracking ctc file line by line

    Args:
        file (Path): File to ctc tracking txt file

    Returns:
        list[dict]: List of the individual lines with a dict[id, start_frame, end_frame, parent_id] containing the tracking information
    """

    # compile regex to match the file
    regex = re.compile(
        r"^(?P<id>\d+) (?P<start_frame>\d+) (?P<end_frame>\d+) (?P<parent_id>\d+)"
    )

    with open(str(file), encoding="utf-8") as input_file:
        data = []
        for line in input_file.readlines():
            m = regex.match(line)

            if m is None:
                # no regex match for this line
                continue

            # add the extracted data
            data.append(m.groupdict())

    return data


def tracking_to_graph(data: list[dict]) -> nx.DiGraph:
    """Populates a ctc tracking into a full tracking lineage where every detection has its own node with a unique id based on (ctc_id, frame)

    Args:
        data (list[dict]): Output of :func:`read_ctc_tracking`

    Returns:
        nx.DiGraph: A lineage graph where every detection has its unique node (id, frame) and the edges represent the linking
    """

    graph = nx.DiGraph()

    # go through every ctc line
    for item in data:
        # iterate every frame for that the cell track exists
        for frame in range(int(item["start_frame"]), int(item["end_frame"]) + 1):
            # add node with unique id (ctc_id, frame)
            graph.add_node((item["id"], frame), frame=frame)
            # add non-division links
            if graph.has_node((item["id"], frame - 1)):
                graph.add_edge((item["id"], frame - 1), (item["id"], frame))

    # add division links
    for item in data:
        if int(item["parent_id"]) != 0:

            # get the time it divides
            # split_frame = int(item["start_frame"]) - 1

            # extract source and target
            source_candidates = list(
                # pylint: disable=cell-var-from-loop
                filter(lambda n: n[0] == item["parent_id"], graph.nodes)
            )
            latest_source = np.argmax(list(map(lambda n: n[1], source_candidates)))
            source = source_candidates[
                latest_source
            ]  # (item["parent_id"], split_frame)
            target = (item["id"], int(item["start_frame"]))

            if not graph.has_node(source):
                print("Error")

            assert graph.has_node(source), f"Source: {source}"
            assert graph.has_node(target), f"Target: {target}"

            # add the edge
            graph.add_edge(source, target)

    return graph
