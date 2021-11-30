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
from acia.segm.omero.storer import OmeroRoISource

from acia.segm.omero.utils import list_images_in_dataset
from acia.segm.output import VideoExporter

#logging.basicConfig(level=logging.DEBUG)

from shapely.geometry import Polygon
from centerline.geometry import Centerline

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

    image_id = 1351

    serverUrl = 'ibt056'
    username = 'bwollenhaupt'
    password = getpass.getpass(f'Password for {username}@{serverUrl}: ')

    omero_cred = {
        'username': username,
        'serverUrl': serverUrl,
        'password': password
    }

    result = {}

    logging.info("Connect to omero...")
    ors = OmeroRoISource(image_id, **omero_cred)

    cell_counts = []
    all_lengths = []
    all_areas = []

    result = process_map(compute_overlay_stats, ors, max_workers=16, chunksize=5)
    for frame_result in result:
        cell_count, lengths, areas = frame_result

        cell_counts.append(cell_count)
        all_lengths.append(lengths)
        all_areas.append(areas)

    max_length = np.max(list(itertools.chain(*all_lengths)))
    max_area = np.max(list(itertools.chain(*all_areas)))


    exp_func = lambda t,a,b: a * np.exp(b*t)
    x = np.array(range(len(cell_counts)))
    y = np.array(cell_counts)
    popt, pcov = fit = scipy.optimize.curve_fit(exp_func, x, y, p0=[1.,1. / len(cell_counts)])

    sq_loss = np.sum((exp_func(x, *popt) - y)**2)
    print(fit)
    print(f"Squared loss: {sq_loss:.3f}")


    #for overlay in tqdm.tqdm(ors):
    #    cell_count = len(overlay)
    #    cell_counts.append(cell_count)
    #
    #    lengths = []
    #    areas = []
    #    for cont in overlay:
    #        length, area = length_and_area(cont)
    #        lengths.append(length)
    #        areas.append(area)
    #
    #    all_lengths.append(lengths)
    #    all_areas.append(area)

    df = pd.DataFrame(np.array(cell_counts), columns=['count'])
    df.to_csv(osp.join(basepath, 'counts.csv'))

    with VideoExporter(osp.join(basepath, 'area_distributions.avi'), 3) as ve:
        for frame, frame_areas in enumerate(tqdm.tqdm(all_areas)):
            fig, ax = plt.subplots(1,1)

            ax.hist(frame_areas, bins=50)
            ax.set_xlabel('area in pixel^2')
            ax.set_ylabel('cell count')
            fig.suptitle(f'Frame: {frame}')
            ax.set_xlim((0, max_area * 1.05))
        

            # add to video
            plt.savefig(osp.join('results', 'area_dist.png'))
            plt.close('all')
            img = cv2.imread(osp.join('results', 'area_dist.png'))
            ve.write(image=img)

    with VideoExporter(osp.join(basepath, 'length_distributions.avi'), 3) as ve:
        for frame, frame_lengths in enumerate(tqdm.tqdm(all_lengths)):
            fig, ax = plt.subplots(1,1)

            ax.hist(frame_lengths, bins=50)
            ax.set_xlabel('length in pixel')
            ax.set_ylabel('cell count')
            fig.suptitle(f'Frame: {frame}')
            ax.set_xlim((0, max_length * 1.05))
        

            # add to video
            plt.savefig(osp.join('results', 'length_dist.png'))
            plt.close('all')
            img = cv2.imread(osp.join('results', 'length_dist.png'))
            ve.write(image=img)


    fig, ax = plt.subplots(1,1)

    ax.plot(cell_counts, label="Real")
    N_0, k = popt
    y = list(map(lambda t: N_0 * np.exp(k*t), range(len(cell_counts))))
    ax.plot(y, label=f"Fit(N_0={N_0:.2f}, k={k:.5f}")

    ax.set_yscale('log')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Cell Count [log]')
    ax.legend()
    fig.suptitle(f'Cell Count over time, Total Count: {np.sum(cell_counts)}')
    plt.tight_layout()
    plt.savefig(osp.join(basepath, "cell_count.png"))

    area_means = []
    length_means = []

    for frame, (lengths, areas) in enumerate(zip(all_lengths, all_areas)):
        area_means.append(np.mean(areas))
        length_means.append(np.mean(lengths))

    fig, ax = plt.subplots(1,1)
    ax.plot(area_means)
    ax.set_xlabel('frame')
    ax.set_ylabel('mean cell area [$pixel^2$]')
    fig.suptitle('Mean Cell Area over time')
    plt.savefig(osp.join(basepath, "mean_cell_area.png"))
    
    fig, ax = plt.subplots(1,1)
    ax.plot(length_means)
    ax.set_xlabel('frame')
    ax.set_ylabel('mean cell length [pixel]')
    fig.suptitle('Mean Cell Length over time')
    plt.savefig(osp.join(basepath, "mean_cell_length.png"))

