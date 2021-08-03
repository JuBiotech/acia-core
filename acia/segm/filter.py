import tqdm
import shapely
from shapely.validation import make_valid
from acia.base import Overlay


class NMSFilter:

    @staticmethod
    def filter(overlay: Overlay, iou_thr=0.1, mode='iou') -> Overlay:
        prefiltered_contours = [cont for cont in overlay.contours if len(cont.coordinates) >= 3]

        # sort contours by their score (lowest first)
        sorted_contours = sorted(prefiltered_contours, key=lambda c: c.score)
        # make (valid) shapely polygons
        polygons = [make_valid(shapely.geometry.polygon.Polygon(contour.coordinates)) for contour in sorted_contours]

        # for i, poly in enumerate(polygons):
        #    if not isinstance(poly, shapely.geometry.polygon.Polygon):
        #        sorted_geoms = sorted(polygons, key=lambda p: -p.area)
        #        polygons[i] = sorted_geoms[i]
        #
        #        if isinstance(polygons[i], shapely.geometry.MultiPolygon):
        #            polygons[i] = sorted(polygons[i].geoms, keay=lambda p: -p.area)[0]

        keep_list = []

        for i, p_i in tqdm.tqdm(enumerate(polygons)):
            keep = True
            for j, p_j in enumerate(polygons[i + 1:]):
                index = j + i + 1
                if sorted_contours[i].frame != sorted_contours[index].frame:
                    continue

                # zero area stuff is not considered
                if p_i.area <= 0:
                    keep = False
                    break

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
