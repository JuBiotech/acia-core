""" Module to convert tracking formats """

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from acia.base import Contour, Overlay


def parse_simple_tracking(file_content: str) -> tuple[Overlay, nx.DiGraph]:
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


def load_ctc_tracklet_graph(file: Path):
    colnames = ["label", "t_start", "t_end", "parent"]

    ctc_df = pd.read_csv(
        file,
        names=colnames,
        header=None,
        dtype={"label": "int", "t_start": "int", "t_end": "int", "parent": "int"},
        delim_whitespace=" ",
    )

    tracklet_graph = nx.DiGraph()

    for _, row in ctc_df.iterrows():
        label, t_start, t_end, parent_label = row.to_list()
        tracklet_graph.add_node(label, start_frame=t_start, end_frame=t_end)

        if parent_label != 0:
            tracklet_graph.add_edge(parent_label, label)

    return tracklet_graph


def ctc_track_graph(ov: Overlay, tracklet_graph: nx.DiGraph):
    """Computes the ctc track graph (every cell detection is a node) based on cell detections (overlay) and the tracklet graph (every tracklet is one node).

    Hint: overlay labels and tracklet_graph node ids need to align.

    Args:
        ov (Overlay): _description_
        tracklet_graph (nx.DiGraph): _description_

    Returns:
        _type_: _description_
    """

    track_graph = nx.DiGraph()

    # add all the nodes
    for cont in ov:
        track_graph.add_node(cont.id)

    tracklets = {}
    for cont in ov:
        tracklets[cont.label] = tracklets.get(cont.label, []) + [cont]

    for tracklet_label in tracklets:
        tracklets[tracklet_label] = sorted(
            tracklets[tracklet_label], key=lambda c: c.frame
        )

    for tracklet_label, tracklet_nodes in tracklets.items():

        # add tracklet edges
        for contA, contB in zip(tracklet_nodes, tracklet_nodes[1:]):
            track_graph.add_edge(contA.id, contB.id)

        for pred_label in tracklet_graph.predecessors(tracklet_label):
            track_graph.add_edge(tracklets[pred_label][-1].id, tracklet_nodes[0].id)

        for succ_label in tracklet_graph.successors(tracklet_label):
            track_graph.add_edge(tracklet_nodes[-1].id, tracklets[succ_label][0].id)

    return track_graph


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
