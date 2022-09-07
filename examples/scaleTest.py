"""Test for applying pixel scale to contour"""

import getpass
import logging
import os
import os.path as osp

import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

from acia.base import Overlay
from acia.segm.omero.storer import OmeroRoISource
from acia.segm.utils import length_and_area

# logging.basicConfig(level=logging.DEBUG)


def compute_overlay_stats(overlay: Overlay):
    cell_count = len(overlay)

    lengths = []
    areas = []
    for cont in overlay:
        length, area = length_and_area(cont)
        lengths.append(length)
        areas.append(area)

    return cell_count, lengths, areas


if __name__ == "__main__":

    basepath = osp.join("results", "cyano")
    os.makedirs(basepath, exist_ok=True)

    image_id = 260

    serverUrl = "ibt056"
    username = "root"
    password = getpass.getpass(f"Password for {username}@{serverUrl}: ")

    omero_cred = {"username": username, "serverUrl": serverUrl, "password": password}

    result = {}

    logging.info("Connect to omero...")
    ors = OmeroRoISource(image_id, **omero_cred, scale="MICROMETER")

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
    ax.set_yscale("log")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cell Count [log]")

    # print(all_areas)

    plt.plot(cell_counts)
