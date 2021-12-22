import getpass
import logging
from omero.gateway import BlitzGateway
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os.path as osp
import os
import cv2
import tqdm
import itertools
import scipy
from tqdm.contrib.concurrent import process_map
from acia import base

from acia.base import Contour, Overlay
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource

from acia.segm.omero.utils import list_images_in_dataset
from acia.segm.output import VideoExporter

#logging.basicConfig(level=logging.DEBUG)

from shapely.geometry import Polygon

def pairwise_distances(points):
    distances = []

    if len(points) == 0:
        return distances

    for a,b in zip(points, points[1:]):
        distances.append(np.linalg.norm(a-b))

    return distances

def length_and_area(contour: Contour):
    polygon = Polygon(contour.coordinates)
    #centerline = Centerline(polygon)
    length = np.max(pairwise_distances(np.array(polygon.minimum_rotated_rectangle.exterior.coords)))
    return length, polygon.area

def compute_overlay_stats(overlay: Overlay):
    cell_count = len(overlay)

    lengths = []
    areas = []
    for cont in overlay:
        length, area = length_and_area(cont)
        lengths.append(length)
        areas.append(area)

    return cell_count, lengths, areas

if __name__ == '__main__':

    basepath = osp.join('results', 'cyano')
    os.makedirs(basepath, exist_ok=True)

    image_id = 260

    serverUrl = 'ibt056'
    username = 'root'
    password = getpass.getpass(f'Password for {username}@{serverUrl}: ')

    omero_cred = {
        'username': username,
        'serverUrl': serverUrl,
        'password': password
    }

    result = {}

    logging.info("Connect to omero...")
    ors = OmeroRoISource(image_id, **omero_cred, scale = "MICROMETER")

    ors.printPixelSize()

    cell_counts = []
    all_lengths = []
    all_areas = []

    result = process_map(compute_overlay_stats, ors, max_workers=1, chunksize=5)
    for frame_result in result:
        cell_count, lengths, areas = frame_result

        cell_counts.append(cell_count)
        all_lengths.append(lengths)
        all_areas.append(areas)

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Cell Count [log]')

    #print(all_areas)

    plt.plot(cell_counts)
