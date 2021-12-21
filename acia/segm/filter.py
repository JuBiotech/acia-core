from typing import Tuple
from numpy.lib.polynomial import poly
import tqdm
import shapely
from shapely.validation import make_valid
from acia.base import Overlay
from shapely.geometry import Polygon
from rtree import index


def bbox_to_rectangle(bbox: Tuple[float]):
    minx, miny, maxx, maxy = bbox
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])


class NMSFilter:

    @staticmethod
    def filter(overlay: Overlay, iou_thr=0.1, mode='iou') -> Overlay:
        prefiltered_contours = [cont for cont in overlay.contours if len(cont.coordinates) >= 3]

        # sort contours by their score (lowest first)
        sorted_contours = sorted(prefiltered_contours, key=lambda c: c.score)
        # make (valid) shapely polygons
        polygons = [make_valid(shapely.geometry.polygon.Polygon(contour.coordinates)) for contour in sorted_contours]

        keep_list = []

        # build an rtree with bounding boxes
        idx = index.Index()
        for i, p_i in enumerate(polygons):
            minx, miny, maxx, maxy = p_i.bounds
            left = minx
            right = maxx
            top = maxy
            bottom = miny
            idx.insert(i, (left, bottom, right, top))

        for i, p_i in tqdm.tqdm(enumerate(polygons), total=len(polygons)):
            keep = True

            # zero area stuff is not considered
            if p_i.area <= 0:
                keep = False
                keep_list.append(keep)
                continue

            # get the intersection candidates by querying the rtree (overlapping bboxes)
            minx, miny, maxx, maxy = p_i.bounds
            left = minx
            right = maxx
            top = maxy
            bottom = miny
            candidate_idx_list = idx.intersection((left, bottom, right, top))

            candidate_idx_list = list(filter(lambda index: index > i and sorted_contours[i].frame == sorted_contours[index].frame, candidate_idx_list))

            # for those candidates we will compute the intersections in details
            for j in candidate_idx_list:
                p_j = polygons[j]

                # compute iou
                if mode == 'i':
                    iou = p_i.intersection(p_j).area / p_i.area  # (p_i.union(p_j).area)
                elif mode == 'iou':
                    iou = p_i.intersection(p_j).area / (p_i.union(p_j).area)
                # compare to threshold
                if iou >= iou_thr:
                    # if exceeding iou drop this cell detection
                    # print("iou: %.2f" % iou)
                    keep = False
                    break

            keep_list.append(keep)

        overlay = Overlay([cont for i, cont in enumerate(sorted_contours) if keep_list[i]])

        return overlay


class SizeFilter:

    @staticmethod
    def filter(overlay: Overlay, min_area, max_area) -> Overlay:
        """Filter an overlay based on contour sizes

        Args:
            overlay (Overlay): the overlay to filter
            min_area ([type]): minimum area of a contour
            max_area ([type]): maximum area of a contour

        Returns:
            Overlay: the filtered overlay
        """
        contour_shapes = [Polygon(cont.coordinates) for cont in overlay.contours]
        result_overlay = Overlay([])
        for cont, shape in zip(overlay.contours, contour_shapes):
            area = shape.area

            if area > min_area and area < max_area:
                result_overlay.add_contour(cont)

        return result_overlay

