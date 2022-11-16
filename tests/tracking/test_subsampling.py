"""Unit test cases for tracking subsampling"""

import unittest
from typing import List

import networkx as nx
import numpy as np

from acia.base import Contour, Overlay
from acia.tracking import TrackingSource, TrackingSourceInMemory, subsample_tracking


class TestSubsampling(unittest.TestCase):
    """Test the linearization of z and t stacks"""

    @staticmethod
    def linear_track(
        length, frame_start, id_start=None, successors: List[TrackingSource] = None
    ):

        if successors is None:
            successors = []

        if id_start is None:
            id_start = 0
            for succ in successors:
                id_start = np.max([id_start, *[cont.id for cont in succ.overlay]]) + 1

        ids = [id_start + i for i in range(length)]
        frames = [frame_start + i for i in range(length)]

        overlay = Overlay(
            [Contour(None, -1, frame, id) for id, frame in zip(ids, frames)]
        )
        tracking_graph = nx.DiGraph()
        tracking_graph.add_nodes_from([cont.id for cont in overlay])
        tracking_graph.add_edges_from(
            [(a.id, b.id) for a, b in zip(overlay.contours, overlay.contours[1:])]
        )

        linear_ts = TrackingSourceInMemory(overlay, tracking_graph)

        for succ in successors:
            # merge the two trackings
            linear_ts.merge(succ)

            # add edge between last and first overlay contour
            linear_ts.tracking_graph.add_edge(
                overlay.contours[-1].id, succ.overlay.contours[0].id
            )

        return linear_ts

    def test_linear(self):
        """Test subsampling of a non-dividing contour"""

        # create tracking source
        tsim = TestSubsampling.linear_track(10, 0)

        # subsample tracking
        tr_source = subsample_tracking(tsim, 2)

        self.assertEqual(tr_source.tracking_graph.number_of_nodes(), 5)
        self.assertEqual(tr_source.tracking_graph.number_of_edges(), 4)

        self.assertSetEqual(
            set(map(lambda c: c.id, tr_source.overlay)), {0, 2, 4, 6, 8}
        )

    def test_division(self):
        """Test subsampling of a dividing contour"""

        # create tracking source
        tsim = TestSubsampling.linear_track(
            5,
            0,
            successors=[
                TestSubsampling.linear_track(3, 5, id_start=0),
                TestSubsampling.linear_track(5, 5, id_start=3),
            ],
        )

        # subsample tracking
        tr_source = subsample_tracking(tsim, 3)

        self.assertEqual(tr_source.tracking_graph.number_of_nodes(), 5)
        self.assertEqual(tr_source.tracking_graph.number_of_edges(), 4)
        self.assertSetEqual(
            set(map(lambda c: c.id, tr_source.overlay)), {8, 11, 1, 4, 7}
        )
        self.assertEqual(tr_source.tracking_graph.out_degree(11), 2)
