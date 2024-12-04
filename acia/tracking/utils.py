"""Utilities for tracking
"""

import networkx as nx

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
