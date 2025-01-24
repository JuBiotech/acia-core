"""Tracking Processor definition"""

import networkx as nx

from acia.base import ImageSequenceSource, Overlay


class TrackingProcessor:
    """Base class for tracking processors"""

    def __call__(
        self, images: ImageSequenceSource, segmentation: Overlay
    ) -> tuple[Overlay, nx.DiGraph, nx.DiGraph]:
        """Perform cell tracking

        Args:
            images (ImageSequenceSource): 2D+t image sequence
            segmentation (Overlay): Segmentation for this image sequence

        Returns:
            tuple[Overlay, nx.DiGraph, nx.DiGraph]: Returns new overlay, the tracklet graph (every cell cycle is a node) and the tracking graph (every cell detection is a node)
        """
        raise NotImplementedError()
