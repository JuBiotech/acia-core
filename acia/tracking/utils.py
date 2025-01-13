"""Utilities for tracking
"""

from itertools import product

import networkx as nx
import numpy as np

from acia.tracking.output import CTCTrackingHelper


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


def delete_nodes(graph: nx.DiGraph, nodes_to_delete: list) -> nx.DiGraph:
    """Delete nodes while maintaining the connectivity

    Args:
        graph (nx.DiGraph): _description_
        nodes_to_delete (list): _description_

    Returns:
        nx.DiGraph: _description_
    """
    for node in nodes_to_delete:
        preds = list(graph.predecessors(node))
        succs = list(graph.successors(node))

        for p, s in product(preds, succs):
            graph.add_edge(p, s)

        graph.remove_node(node)

    return graph


def subsample_lineage(lineage: nx.DiGraph, subsampling_factor: int) -> nx.DiGraph:
    """Subsample lineage by only takeing nodes in every n-th (subsampling_factor) frame. Connectivity is maintained.

    Args:
        lineage (nx.DiGraph): lineage graph (needs the frame attributes)
        subsampling_factor (int): n-th frame will taken into account

    Returns:
        nx.DiGraph: Returns the pruned lineage only containing nodes of every n-th frame
    """

    # copy the lineage
    lineage = lineage.copy()

    # get all the frames in the lineage
    frames = list(sorted(np.unique([lineage.nodes[n]["frame"] for n in lineage.nodes])))

    # compute what frames to keep (every n-th)
    keep_frames = set(frames[::subsampling_factor])

    # create a list of nodes that are not inside the selected frames
    del_nodes = [
        n for n in lineage.nodes if lineage.nodes[n]["frame"] not in keep_frames
    ]

    # delete nodes (maintain connectivity)
    new_lineage = delete_nodes(lineage, del_nodes)

    # return the new lineage
    return new_lineage
