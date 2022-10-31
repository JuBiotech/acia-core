"""Tracking module contains all tools to work with tracking formats
"""

from pathlib import Path

import networkx as nx
import numpy as np

from acia.base import Overlay
from acia.tracking.output import CTCTrackingHelper

from .formats import gen_simple_tracking, parse_simple_tracking


class TrackingSource:
    """Base class for tracking information containing segmentation overlay and tracking graph (usually ids of overlay contours)"""

    @property
    def overlay(self) -> Overlay:
        raise NotImplementedError()

    @property
    def tracking_graph(self) -> nx.DiGraph:
        raise NotImplementedError()

    def copy(self) -> "TrackingSource":
        raise NotImplementedError()


class TrackingSourceInMemory(TrackingSource):
    """Tracking Source stored in memory"""

    def __init__(self, overlay: Overlay, tracking_graph: nx.DiGraph):
        super().__init__()
        self.__overlay = overlay
        self.__tracking_graph = tracking_graph

    @property
    def overlay(self) -> Overlay:
        return self.__overlay

    @property
    def tracking_graph(self) -> nx.DiGraph:
        return self.__tracking_graph

    def copy(self) -> "TrackingSourceInMemory":
        return TrackingSourceInMemory(
            Overlay(list(self.overlay)), self.tracking_graph.copy()
        )

    def merge(self, tr_source: TrackingSource):
        tr_source = tr_source.copy()

        self.__overlay = Overlay(self.overlay.contours + tr_source.overlay.contours)
        self.__tracking_graph = nx.compose(
            self.tracking_graph, tr_source.tracking_graph
        )

        return self


class SimpleTrackingSource(TrackingSourceInMemory):
    """Tracking Source based on simple tracking json format"""

    def __init__(self, file_content: str):
        super().__init__(*parse_simple_tracking(file_content))

    @staticmethod
    def from_file(file_path: Path) -> "SimpleTrackingSource":
        """Loads segmentation and tracking from simple tracking json format

        Args:
            file_path (Path): path to the simple tracking file

        Returns:
            SimpleTrackingSource: the loaded simple tracking file
        """
        with open(file_path, encoding="utf-8") as input_file:
            return SimpleTrackingSource(input_file.read())

    def store(self, file_path: Path):
        """Saves simple tracking json format

        Args:
            file_path (Path): file name to save
        """
        with open(file_path, "w", encoding="utf-8") as output_file:
            output_file.write(gen_simple_tracking(self.overlay, self.tracking_graph))


def subsample_tracking(
    tracking: TrackingSource, subsampling_factor: int
) -> TrackingSource:
    """Subsample the tracking source

    Args:
        tracking (TrackingSource): tracking source to subsample
        subsampling_factor (int): subsampling factor defining the step of frames. 1 means no subsampling. 2 means every second frame, ...

    Raises:
        ValueError: when wrong subsampling factor is chosen

    Returns:
        TrackingSource: subsampled tracking source
    """

    if subsampling_factor < 1:
        raise ValueError("Please chose a subsampling factor >= 1")

    # extract information from source
    overlay = tracking.overlay
    tracking_graph = tracking.tracking_graph

    # subsample frames
    subsampled_frames = set(
        np.arange(overlay.numFrames(), step=subsampling_factor, dtype=np.int32)
    )

    frame_lookup = {
        old_frame: new_frame
        for new_frame, old_frame in zip(
            range(len(subsampled_frames)), sorted(subsampled_frames)
        )
    }

    # and create overlay with remaining contours
    subsampled_overlay = Overlay(
        list(filter(lambda cont: cont.frame in subsampled_frames, overlay)),
        frames=list(range(len(subsampled_frames))),
    )

    for cont in subsampled_overlay:
        cont.frame = frame_lookup[cont.frame]

    # copy tracking graph
    subsampled_graph = tracking_graph.copy()

    # compute the set of segment ids we have to remove
    subsampled_overlay_ids = {cont.id for cont in subsampled_overlay}
    nodes_to_remove = set(tracking_graph.nodes).difference(
        subsampled_overlay_ids
    )  # [node for node in nx.topological_sort(tracking_graph) if node not in subsampled_overlay_ids]

    # loop over all these segments to remove
    for node in nodes_to_remove:
        # get parents and children
        parents = list(subsampled_graph.predecessors(node))
        children = list(subsampled_graph.successors(node))

        # for every edge: (parent --> node --> child) insert edge: (parent --> child) into the subsampled graph
        for parent in parents:
            for child in children:
                # connect parent to children
                subsampled_graph.add_edge(parent, child)

        subsampled_graph.remove_node(node)

    # make sure that we have still all contours of the overlay in our tracking
    assert len(set(subsampled_overlay_ids).difference(set(subsampled_graph.nodes))) == 0

    # return the subsampled tracking source
    return TrackingSourceInMemory(subsampled_overlay, subsampled_graph)


def life_cycle_lineage(tr_graph: nx.DiGraph) -> nx.DiGraph:
    """Compresses populated lineage to life cycle lineage (one node per cell cycle)

    Args:
        tr_graph (nx.DiGraph): populated tracking graph

    Returns:
        nx.DiGraph: Life cycle lineage with cell cylces as nodes
    """

    # compute the life-cycles of individual cells
    life_cycles = CTCTrackingHelper.compute_life_cycles(tr_graph)
    # create lookup (cont id --> life cycle index)
    life_cycle_lookup = CTCTrackingHelper.create_life_cycle_lookup(life_cycles)
    # contour_lookup = {cont.id: cont for cont in overlay}

    lc_graph = nx.DiGraph()

    # add all the nodes
    lc_graph.add_nodes_from(range(len(life_cycles)))

    # set the "cycle" property to contain the populated life cycle nodes
    for i, life_cycle in enumerate(life_cycles):
        lc_graph.nodes[i]["cycle"] = life_cycle

    # iterate over life_cycles
    for lc_id, lc in enumerate(life_cycles):
        start = lc[0]

        # extract parents from populated tracking
        parents = tr_graph.predecessors(start)

        for parent in parents:
            # get the parent life_cycle
            parent_lc_id = life_cycle_lookup[parent]

            # establish an edge between parent and child
            lc_graph.add_edge(parent_lc_id, lc_id)

    # set "start_frame" and "end_frame" for every node in the life cycle graph
    for node in lc_graph:
        lc = lc_graph.nodes[node]["cycle"]

        lc_graph.nodes[node]["start_frame"] = tr_graph.nodes[lc[0]]["frame"]
        lc_graph.nodes[node]["end_frame"] = tr_graph.nodes[lc[-1]]["frame"]

    return lc_graph
