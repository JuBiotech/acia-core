""" Test acia base functionality"""

import time
import unittest

import numpy as np

from acia.base import Contour, Instance
from acia.segm.formats import load_ctc_segmentation_native
from acia.tracking.formats import ctc_track_graph, load_ctc_tracklet_graph


class TestContour(unittest.TestCase):
    """Test contour functionality"""

    def test_center(self):
        contour = [[0, 0], [1, 0], [1, 1], [0, 1]]

        self.assertTrue(contour is not None)

        # simple contour
        np.testing.assert_array_equal(
            Contour(contour, 0, 0, 0).center, np.array([0.5, 0.5], dtype=np.float32)
        )

        # unequal point sampling
        contour = [[0, 0], [0.5, 0], [1, 0], [1, 1], [0, 1]]
        np.testing.assert_array_equal(
            Contour(contour, 0, 0, 0).center, np.array([0.5, 0.5], dtype=np.float32)
        )

    def test_rasterization(self):
        """Make sure that contour to mask rasterization preserves area"""

        contours = [[[0, 0], [10, 0], [10, 10], [0, 10]]]

        for coordinates in contours:
            cont = Contour(coordinates, -1, 0, -1)
            mask = cont.toMask(40, 40)

            self.assertEqual(cont.area, np.sum(mask))


class TestInstance(unittest.TestCase):
    """Test contour functionality"""

    def test_center(self):
        mask = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 3, 3, 0],
                [0, 3, 3, 3, 3],
                [0, 2, 2, 0, 0],
            ],
            dtype=np.uint8,
        )

        instance = Instance(mask, 0, 3)

        self.assertEqual(instance.area, 6)

        poly = instance.polygon

        self.assertEqual(poly.area, 6)

        instance = Instance(mask, 0, 2)
        self.assertEqual(instance.area, 2)

        poly = instance.polygon
        self.assertEqual(poly.area, 2)

    def test_load_native(self):
        print("native")
        t_start = time.time()
        ov = load_ctc_segmentation_native("00_GT/SEG")
        t_end = time.time()
        print(len(ov))
        print("Loading time (s) ", t_end - t_start)

        t_start = time.time()
        _ = [cont.area for cont in ov]
        t_end = time.time()

        ids = [cont.id for cont in ov]

        self.assertEqual(len(ids), len(set(ids)))

        print("Area computation time (s)", t_end - t_start)

    def test_load_tracking_native(self):

        print("native")
        t_start = time.time()
        ov = load_ctc_segmentation_native("00_GT/TRA")
        t_end = time.time()
        print(len(ov))
        print("Load overlay time (s) ", t_end - t_start)

        t_start = time.time()
        tracklet_graph = load_ctc_tracklet_graph("00_GT/TRA/man_track.txt")
        t_end = time.time()
        print("Load tracking time (s) ", t_end - t_start)

        labels = np.unique([cont.label for cont in ov])
        nodes = set(tracklet_graph.nodes)

        # make sure that the labels from the tracking annotations are equal to the tracklet nodes
        self.assertEqual(set(labels), nodes)

        # make the tracking graph
        track_graph = ctc_track_graph(ov, tracklet_graph)

        self.assertEqual(len(ov), track_graph.number_of_nodes())

        for node_id in track_graph.nodes:
            for succ_id in track_graph.successors(node_id):
                self.assertTrue(ov[node_id].frame <= ov[succ_id].frame - 1)

            for pred_id in track_graph.predecessors(node_id):
                self.assertTrue(ov[node_id].frame >= ov[pred_id].frame + 1)


if __name__ == "__main__":
    unittest.main()
