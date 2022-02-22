from mmdet.apis import init_detector, inference_detector
import numpy as np
import cv2
import mmcv
import logging
from shapely.geometry import Polygon, LineString
import rtree
from typing import List
import time
import argparse
import tifffile
import os
import tqdm.auto as tqdm
from mmcv.runner import wrap_fp16_model
import torch
import roifile

logger = logging.getLogger(__name__)


def contour_from_mask(mask, score_threshold):
    '''
        Estimate largest contour from pixel-wise mask
    '''
    contours, hierarchy = cv2.findContours(np.where(mask > score_threshold, 1, 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # select largest contour
    selected_contour = []
    for cont in contours:
        if len(cont) > len(selected_contour):
            selected_contour = cont

    return np.squeeze(selected_contour)


def prepare_contours(bboxes, segm_result, labels, offset_x=0, offset_y=0, seg_score_threshold=0.3):
    offset = np.array([offset_x, offset_y])

    all_contours = []
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)

        for seg in segms:
            seg = seg.astype(np.float32)

            # Creating kernel
            # kernel = np.ones((3, 3), np.uint8)
            # find contours with cv2
            contours, hierarchy = cv2.findContours(np.where(seg > seg_score_threshold, 1, 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            selected_contour = []
            for cont in contours:
                if len(cont) > len(selected_contour):
                    selected_contour = cont

            cont_data = [(np.ceil(cont).astype(np.int32).squeeze(axis=1) + offset[None, :]) for cont in [selected_contour] if len(cont) > 0]  # [[str(x), str(y)] for cont in contours for x,y in cont]

            logging.info("Num contours: %d" % len(contours))

            all_contours.append(cont_data)

    return all_contours


def postprocess(output_data, model, offset_x=0, offset_y=0, contours=False):
    segm_result = None

    if model.with_mask:
        bbox_result, segm_result = output_data
    else:
        bbox_result = output_data

    output_contours = (segm_result is not None) and contours

    bboxes = np.vstack(bbox_result)

    # print("Num bounding boxes:")

    # identify the labels of the detected boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    # prepare contours if necessary
    if output_contours:
        contours = prepare_contours(bboxes, segm_result, labels, offset_x, offset_y)
        logging.info("Num contours: %d" % len(contours))
    else:
        contours = [[]] * len(bboxes)

    result = []

    for box, label, contour, segm in zip(bboxes, labels, contours, mmcv.concat_list(segm_result)):
        if output_contours and len(contour) == 0:
            # skip this detection
            continue

        box_coords = list(map(float, box[:4]))
        score = float(box[4])
        label_name = model.CLASSES[label]

        # print("Class '%s' detected at %s with score %.2f" % (label_name, box_coords, score))

        result_dict = {'label': label_name, 'bbox': box_coords, 'score': score, 'mask': segm}

        # sort descending
        if output_contours:
            result_dict['contours'] = sorted([{'x': cont[:, 0], 'y': cont[:, 1]} for cont in contour], key=lambda cont: -len(cont['x']))

        result.append(result_dict)  # 'contours': contour

    return result


def tile_touch_filter(detection, image_tile_poly: LineString, threshold=10):
    '''
        returns False iff detections it too close (<threshold) to the boundaries and likely to be restricted by them

        detection: the detection dict
        image_tile_poly: Line string of the image tile boundaries
        threshold: minimum allowed distance to the boundaries
    '''

    contour = detection['contours'][0]
    detection_poly = Polygon(zip(contour['x'], contour['y']))

    distance = detection_poly.distance(image_tile_poly)

    # print(distance)

    return distance >= threshold


def inference(image, model, offset_x=0, offset_y=0):
    # inference on tile
    raw_tile_results = inference_detector(model, image)

    # postprocess
    tile_results = postprocess(raw_tile_results, model, offset_x=offset_x, offset_y=offset_y)

    return tile_results


def tiled_inference(image, model, x_shift=256 - 128, y_shift=256 - 128, tile_width=256, tile_height=256, pd=25):
    '''
        Execute inference in a tiled fashion

        x_shift: shift on the x-axis for every image slot
        y_shift: shift on the y-axis for every image slot
        tile_width: width of the image tile
        tile_height: height of the image tile

        TODO: When tiles align with image borders, we should not do tile touch filtering
    '''

    # get the image dimensions
    height, width = image.shape[:2]

    x_start = 0
    y_start = 0

    all_detections = []

    # padding the image (rgb)
    padding_size = pd
    padded_image = np.zeros((height + 2 * padding_size, width + 2 * padding_size, 3), dtype=np.uint8)
    padded_image[padding_size:padding_size + height, padding_size:padding_size + width] = image
    orig_image = image
    image = padded_image

    height, width = image.shape[:2]

    # iterate over top coordinate of tile
    ys = list(range(max(1, 1 + int(np.ceil((height - tile_height) / y_shift)))))
    for iY in ys:
        y = y_start + iY * y_shift
        # iterate over left coordinate of tile
        xs = list(range(max(1, 1 + int(np.ceil((width - tile_width) / x_shift)))))
        for iX in xs:
            x = x_start + iX * x_shift

            # print(x,y)

            # compute the lower right coordinates of the tile
            y_end = min(height, y + tile_height)
            x_end = min(width, x + tile_width)

            # print(x_end, y_end)
            # print(y_end-y, x_end -x)
            # get the image tile
            image_tile = image[y:y_end, x:x_end]
            # print(image_tile.shape)
            # zero padding to constant tile size (otherwise we get devision errors)
            const_tile_format = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            const_tile_format[:y_end - y, :x_end - x] = image_tile

            # print(const_tile_format.shape)

            tile_results = inference(const_tile_format, model, x, y)

            tile_results = list(filter(lambda det: np.sum(det['mask']) >= 3, tile_results))

            if len(tile_results) > 0:
                filter_mask = mask_nms(np.stack([det['mask'] for det in tile_results]), np.stack([det['bbox'] for det in tile_results]), np.stack([det['score'] for det in tile_results]))

                tile_results = list(np.array(tile_results)[filter_mask])

            # print(len(tile_results))

            # polygon for the image tile
            # image_tile_poly = LineString([(x, y), (x+tile_width, y), (x+tile_width, y+tile_height), (x, y+tile_height), (x,y)])

            # filter the detections
            #   -> no detections close to the border of the tile schould be considered
            filter_mask = np.ones(len(tile_results), dtype=np.bool)

            for i, det in enumerate(tile_results):
                det_mask = det['mask']

                row, col = np.nonzero(det_mask)

                miny = np.min(row)
                maxy = np.max(row)
                minx = np.min(col)
                maxx = np.max(col)

                min_distance = np.min([
                    miny, minx, tile_height - maxy, tile_width - maxx
                ])

                if min_distance < padding_size:
                    filter_mask[i] = False

            tile_results = list(np.array(tile_results)[filter_mask])
            # tile_results += filter(partial(tile_touch_filter, image_tile_poly=image_tile_poly), tile_results)

            new_masks = np.zeros((len(tile_results), *orig_image.shape[:2]), dtype=np.bool)

            # expand masks to full image
            for i, det in enumerate(tile_results):
                new_mask = new_masks[i]

                y_offset = 0
                y_endset = 0
                if iY == 0:
                    y_offset = pd
                if y_end > orig_image.shape[0] + pd:
                    y_endset = y_end - (orig_image.shape[0] + pd)
                x_offset = 0
                x_endset = 0
                if iX == 0:
                    x_offset = pd
                if x_end > orig_image.shape[1] + pd:
                    x_endset = x_end - (orig_image.shape[1] + pd)

                mask_height = (y_end - y) - y_endset
                mask_width = (x_end - x) - x_endset

                new_mask[max(0, y - pd):y - pd + mask_height, max(0, x - pd):x - pd + mask_width] = det['mask'][y_offset:mask_height, x_offset:mask_width]  # [:y_end - y,:x_end - x]
                det['mask'] = new_mask
                det['bbox'] += np.array([x, y, x, y]) - pd

            all_detections += tile_results

    return all_detections


def non_max_supression(all_detections: List[Polygon], iou=0.3):
    '''
        Performing something like non-maximum supression on a list of detections

        TODO: make sure that this corresponds with some paper for nms

        all_detections: all detections found in an image
        iou: intersection over union: if a poly intersects more than that with another poly and it's score is lower it gets discarded.

        returns the filtered list of detections
    '''
    # descending sort
    all_detections = sorted(all_detections, key=lambda det: det['score'])
    polygons = []
    for det in all_detections:
        contour = det['contours'][0]
        xs = contour['x']
        ys = contour['y']

        poly = Polygon(zip(xs, ys))

        if not poly.is_valid:
            print('Invalid polygon!')
            '''x,y = poly.exterior.xy
            #fig = plt.figure()
            #plt.plot(x,y)
            other_poly = poly.buffer(2)
            if isinstance(other_poly, MultiPolygon):
                max_area_index = np.argmax(list(map(lambda g: g.area, other_poly.geoms)))
                other_poly = other_poly.geoms[max_area_index].buffer(-2)

                other_poly = Polygon(np.round(list(zip(*other_poly.exterior.xy))))
            try:
                x,y = other_poly.exterior.xy
            except:
                x,y = poly.exterior.xy
                fig = plt.figure()
                plt.plot(x,y)
                x,y = poly.buffer(2).exterior.xy
                plt.plot(x,y)
                plt.savefig('problem.png')
                plt.close(fig)
            poly=other_poly
            #plt.plot(x,y)
            #plt.savefig('problem.png')
            #plt.close(fig)'''

        polygons.append(poly)

    idx = rtree.index.Index()
    for pos, poly in enumerate(polygons):
        idx.insert(pos, poly.bounds)

    set_remove_indices = set()

    # Loop through each Shapely polygon
    for i, poly in enumerate(polygons):
        score = all_detections[i]['score']
        area = poly.area

        # Merge cells that have overlapping bounding boxes
        for pos in idx.intersection(poly.bounds):
            if pos == i:
                continue

            poly_other = polygons[pos]
            score_other = all_detections[pos]['score']

            # distance = poly.distance(poly_other)
            # intersect = poly.intersects(poly_other)

            # print(poly.is_valid)
            # print(poly_other.is_valid)

            poly_other = poly_other.buffer(0)
            # print(poly_other.is_valid)

            # compute intersection
            intersect_poly = poly.intersection(poly_other)
            intersect_area = intersect_poly.area

            if score < score_other and intersect_area / area > iou:
                # do not take poly
                set_remove_indices.add(i)
                break

    return list(map(lambda idet: idet[1], filter(lambda idet: not idet[0] in set_remove_indices, enumerate(all_detections))))


def np_vec_no_jit_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou


def torch_vec_no_jit_iou(boxes1, boxes2):
    x11, y11, x12, y12 = torch.chunk(boxes1, 4, dim=1)
    x21, y21, x22, y22 = torch.chunk(boxes2, 4, dim=1)

    xA = torch.maximum(x11, torch.transpose(x21, 0, 1))
    yA = torch.maximum(y11, torch.transpose(y21, 0, 1))
    xB = torch.minimum(x12, torch.transpose(x22, 0, 1))
    yB = torch.minimum(y12, torch.transpose(y22, 0, 1))

    interArea = torch.maximum((xB - xA + 1), torch.tensor(0, device=boxes1.device)) * torch.maximum((yB - yA + 1), torch.tensor(0, device=boxes1.device))
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + torch.transpose(boxBArea, 0, 1) - interArea)

    return iou


def torch_mask_nms(masks, bboxes, scores, bbox_iou_threshold=.1, mask_iou_threshold=.4, score_threshold=0.1):
    '''
        iou: if intersection between two cells is larger, only take the better scored one
    '''

    device = 'cuda:0'

    masks = torch.tensor(masks, device=device)
    bboxes = torch.tensor(bboxes, device=device)

    # areas = torch.sum(torch.tensor(masks), axis=(1,2)).numpy()

    bbox_iou = torch_vec_no_jit_iou(bboxes, bboxes)

    print(masks.shape)

    # print(masks.nbytes)

    filter_mask = scores >= score_threshold

    scores = torch.tensor(scores, device=device)

    drops = []  # torch.zeros_like(scores, dtype=torch.bool)

    # intersection = masks[None] & np.r
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if not filter_mask[i]:
            continue

        candidate_mask = bbox_iou[i, :] > bbox_iou_threshold

        # area = areas[i]#np.sum(mask)
        intersection = mask[None, :] & masks[candidate_mask]
        joint = mask[None, :] | masks[candidate_mask]

        intersection_areas = torch.sum(intersection, dim=(1, 2))
        joint_areas = torch.sum(joint, dim=(1, 2))

        relative_intersections = intersection_areas / joint_areas

        over_threshold = torch.where(relative_intersections > mask_iou_threshold)

        higher_scored = scores[candidate_mask][over_threshold] > score

        drops.append(~(torch.sum(higher_scored) >= 1).cpu())

        # if drop:
        #    filter_mask[i] = False

        # print(relative_intersections)

    filter_mask = np.array(drops, dtype=np.bool)

    return np.arange(len(masks))[filter_mask]


def mask_nms(masks, bboxes, scores, bbox_iou_threshold=.1, mask_iou_threshold=.4, score_threshold=0.1):
    '''
        iou: if intersection between two cells is larger, only take the better scored one
    '''
    # masks = torch.tensor(masks)
    # bboxes = torch.tensor(bboxes)

    # areas = torch.sum(torch.tensor(masks), axis=(1, 2)).numpy()

    bbox_iou = torch_vec_no_jit_iou(torch.tensor(bboxes), torch.tensor(bboxes)).numpy()

    print(masks.shape)

    print(masks.nbytes)

    filter_mask = scores >= score_threshold
    # intersection = masks[None] & np.r
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if not filter_mask[i]:
            continue

        candidate_mask = bbox_iou[i, :] > bbox_iou_threshold

        # area = areas[i]#np.sum(mask)
        intersection = mask[None, :] & masks[candidate_mask]
        joint = mask[None, :] | masks[candidate_mask]

        intersection_areas = np.sum(intersection, axis=(1, 2))
        joint_areas = np.sum(joint, axis=(1, 2))

        relative_intersections = intersection_areas / joint_areas

        over_threshold = np.where(relative_intersections > mask_iou_threshold)

        higher_scored = scores[candidate_mask][over_threshold] > score

        drop = np.sum(higher_scored) >= 1

        if drop:
            filter_mask[i] = False

        # print(relative_intersections)

    return np.arange(len(masks))[filter_mask]


def prediction(image, model, min_score=0.0, tiling=None):
    # apply tiled inference
    if tiling:
        all_detections = tiled_inference(image, model, **tiling)
        # filter by score
        all_detections = list(filter(lambda det: det['score'] > min_score, all_detections))
        # perform non-max supressions (due to tiling this is needed)
        if len(all_detections) > 0:
            filter_mask = torch_mask_nms(np.stack([det['mask'] for det in all_detections]), np.stack([det['bbox'] for det in all_detections]), np.stack([det['score'] for det in all_detections]),
                                         score_threshold=min_score, mask_iou_threshold=0.6)

            all_detections = list(np.array(all_detections)[filter_mask])
    else:
        all_detections = inference(image, model)
        all_detections = list(filter(lambda det: det['score'] > min_score, all_detections))

        # filter_mask = torch_mask_nms(np.stack([det['mask'] for det in all_detections]), np.stack([det['bbox'] for det in all_detections]), np.stack([det['score'] for det in all_detections]),
        #                score_threshold=0.0, mask_iou_threshold=0.4)

        all_detections = all_detections  # list(np.array(all_detections)[filter_mask])

    return all_detections


def generate_video(input_folder, output_file):
    import ffmpeg

    (
        ffmpeg
        .input('output/*.png', pattern_type='glob', framerate=1)
        .output('output.mp4')
        .run()
    )
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure Prediction')

    parser.add_argument('--weights', dest='weights', type=str,
                        # default='mass_evaluation/me_cascade_mask_rcnn_2_x101_64x4d_False_True_15_True/latest.pth'
                        default='train_work_dirs/train_data/me_htc_2_False_True_15_True_7/latest.pth',
                        # default='work_dirs/htc_best/epoch_15.pth',
                        help='path/to/file.pth containing the newtork weights')
    parser.add_argument('--config', dest='config', type=str,
                        # default='mass_evaluation/me_cascade_mask_rcnn_2_x101_64x4d_False_True_15_True/me_cascade_mask_rcnn_2_x101_64x4d_False_True_15_True.py',
                        default="train_work_dirs/train_data/me_htc_2_False_True_15_True_12/me_htc_2_False_True_15_True_12.py",
                        # default='work_dirs/htc_best/htc_best.py',
                        help='path/to/config.py containing the network architecture definition')

    parser.add_argument('--mode', dest='mode', choices=['mask', 'contour', 'tif', 'roi'], default='contour')
    parser.add_argument('--tiling', dest='tiling', type=bool, default=False, help='Enables tiling image into several smaller images. Especially helpful when many cells are present')
    parser.add_argument('--padding', dest='padding', type=int, default=0, help='Enables pixel padding of the images')
    parser.add_argument('--out-file', type=str, default='output.tif', help="Output file")
    parser.add_argument('input', type=str, default='output.tif', help="Input file")
    parser.add_argument('--bbox', dest='bbox', type=bool, default=True, help="Draw a bounding box around object detections")

    args = parser.parse_args()

    config_file = args.config  # 'mass_evaluation/me_cascade_mask_rcnn_2_x101_64x4d_False_True_15_True/me_cascade_mask_rcnn_2_x101_64x4d_False_True_15_True.py'#'htc_best.py'#'mass_evaluation/me_htc_2_x101_64x4d_True_True_15_True/me_htc_2_x101_64x4d_True_True_15_True.py'#'mask_rcnn_r50_fpn_1x_coco.py'
    input_tif = args.input  # 'input/PHH2.nd2-PHH2.nd2(series6)_rois-1_70_final.tif' #'input/PHH1001.nd2-PHH1001.nd2(series11)_preprocessed_cropped_rois-1_48_final-1.tif' #'input/2021-01-20 E.coli SIMBAL GFP_LB_CGM9_30°C_bat_pos37.tif_primed.tif'#'input/nd059.nd2 - nd059.nd2 (series 81)-80_crop.tif'
    out_file = args.out_file
    path = args.weights  # 'mass_evaluation/me_cascade_mask_rcnn_2_x101_64x4d_False_False_15_False/epoch_15.pth'#'work_dirs/htc_best/latest.pth'#'mass_evaluation/me_htc_2_x101_64x4d_True_True_15_True/latest.pth'#args.weights
    mode = args.mode
    device = 'cuda:0'

    try:
        os.mkdir('output')
    except FileExistsError:
        pass

    print('Loading model...')
    # init a detector
    model = init_detector(config_file, path, device=device,
                          cfg_options={
                              'test_cfg.rpn.nms_thr': 0.7,
                              'test_pipeline[1].scale_factor': 1.,
                              'test_cfg.rcnn.nms.type': 'nms',
                              'test_cfg.rcnn.nms.iou_threshold': 0.4})

    wrap_fp16_model(model)
    # inference the demo image

    print('Loading data...')
    # image = np.array(Image.open('test4.png'))
    input_tif_file = tifffile.TiffFile(input_tif)
    orig_images = images = input_tif_file.asarray()  # tifffile.imread(input_tif)
    imagej_metadata = input_tif_file.imagej_metadata

    new_images = []
    tiling = args.tiling
    padding = args.padding

    # normalize images and perform padding
    for i, image in enumerate(images):
        min_val = np.min(image)
        max_val = np.max(image)
        image = np.floor((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        image = np.pad(image, padding)

        if len(image.shape) > 2:
            image = image[0]

        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], 3, axis=-1)

        new_images.append(image)

    images = np.stack(new_images)

    cv2.imwrite('out.png', images[0])
    # exit(0)

    duration = 0

    rois = []

    for i, image in tqdm.tqdm(enumerate(images)):

        # print(image.shape)

        if len(image.shape) == 2:
            image = image[:, :, None]
            image = np.repeat(image, 3, axis=-1)

        start = time.time()
        all_detections = prediction(image, model, tiling=tiling)
        end = time.time()

        duration += (end - start)

        # cont_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
        all_masks = np.stack([det['mask'] for det in all_detections])

        if mode == 'mask':
            joint_mask = np.any(all_masks, axis=0)
            overlay = joint_mask[:, :, None] * np.array((0, 255, 0), dtype=np.uint8)[None, None]
        elif mode in ['contour', 'tif', 'roi']:
            all_contours = [contour_from_mask(mask, 0.5) for mask in all_masks]
            # drop non-sense contours
            all_contours = list(filter(lambda contour: len(contour) >= 5, all_contours))

            if mode == 'contour':
                overlay = np.zeros_like(image)
                cv2.drawContours(overlay, all_contours, -1, (0, 255, 0), 1)
            else:
                for contour in all_contours:
                    rois.append(roifile.ImagejRoi.frompoints(contour - padding, t=i))
        else:
            print('Erroneous mode')
            exit(1)

        # if args.bbox:
        #    for mask in all_masks:
        #        rows = np.any(mask, axis=1)
        #        cols = np.any(mask, axis=0)
        #        rmin, rmax = np.where(rows)[0][[0, -1]]
        #        cmin, cmax = np.where(cols)[0][[0, -1]]

        #        cv2.drawPoly(overlay, np.array([[cmin, rmin], [cmax, rmin], [cmax, rmax], [cmin, rmax]]), -1, (0, 0, 255))

        if mode not in ['tif', 'roi']:
            cont_image = np.where(overlay > 0, overlay, image)  # cv2.addWeighted(image, 1.0, overlay, 0.6, 0.)
            print("Write image")
            cv2.imwrite('output/%04d.png' % i, cont_image)

    print("Raw prediction duration %.2fs or %.2fs/images" % (duration, duration / len(images)))

    if mode == 'tif':
        filename = args.out_file if not (args.out_file is None) else 'output.tif'  # 'output.tif'
        overlays = [roi.tobytes() for roi in rois]
        # use the original images without padding
        imagej_metadata['Overlays'] = overlays
        tifffile.imsave(filename, orig_images, imagej=True, ijmetadata=imagej_metadata)
        # tifffile.imsave(filename, orig_images, imagej=True, ijmetadata=imagej_metadata)
    elif mode == 'roi':
        # write all rois to a zipfile
        roifile.roiwrite(out_file, rois)
    else:
        generate_video('output', 'output.mp4')

    # inference_detector(model, images)
    # result_data = inference_detector(model, image)

    # print(postprocess(result_data,  model))
