""" Example to estimate the fluorescence error """

import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import tqdm.auto as tqdm
from config import basepath
from pixel_clustering import cluster_pixels

from acia.segm.omero.storer import OmeroSequenceSource
from examples.analysis.fluorescence.area_distributions import compute_area_bounds

# from examples.analysis.fluorescence.config import clustering


def clustering():
    raise NotImplementedError()


def cell_count_bounds(pixel_count, lower_size, upper_size):
    # min and max of cells
    return pixel_count / upper_size, pixel_count / lower_size


def main():

    # omero id of the image (Image ID)
    image_id = 470
    # fluorescence channels you want to monitor (usually 1 is the phase contrast)
    fluorescence_channels = [2, 3]

    # your user credentials for omero
    credentials = dict(
        username="root",
        password="omero",
        serverUrl="ibt056",
    )

    # combine images and rois
    # irs = ImageRoISource(
    oss = OmeroSequenceSource(
        image_id,
        **credentials,
        channels=fluorescence_channels,
        colorList=["FF0000", "00FF00"]
    )
    #    OmeroRoISource(image_id, **credentials)
    # )

    df, transform, kmeans, red_index, green_index = clustering()

    counts = []

    bounds = []
    counts = []

    for frame, image in enumerate(tqdm.tqdm(oss)):
        # predict cluster labels for individual pixels
        predicted_labels = cluster_pixels(image, transform, kmeans)

        f_df = df[df["frame"] == frame]

        red_cell_count = len(f_df[f_df["color"] == "red"])
        green_cell_count = len(f_df[f_df["color"] == "green"])

        red_pixel_count = np.sum(predicted_labels == red_index)
        green_pixel_count = np.sum(predicted_labels == green_index)

        red_bounds = compute_area_bounds(df[df["frame"] == frame], "red")
        green_bounds = compute_area_bounds(df[df["frame"] == frame], "green")

        if not red_bounds is None:
            red_cc_bounds = cell_count_bounds(red_pixel_count, *red_bounds)
        else:
            red_cc_bounds = (0, 1e3)

        if not green_bounds is None:
            green_cc_bounds = cell_count_bounds(green_pixel_count, *green_bounds)
        else:
            green_cc_bounds = (0, 1e3)

        print(red_cc_bounds, green_cc_bounds)
        bounds.append((red_cc_bounds, green_cc_bounds))

        counts.append((red_cell_count, green_cell_count))

    bounds = np.array(bounds)
    counts = np.array(counts)

    fig, ax = plt.subplots(2, 1)

    ax[0].fill_between(
        range(len(bounds)),
        bounds[:, 0, 0],
        bounds[:, 0, 1],
        label="uncertain cell count",
        alpha=0.2,
        color="red",
    )
    ax[0].plot(counts[:, 0], color="red", label="predicted cell count")
    ax[1].fill_between(
        range(len(bounds)),
        bounds[:, 1, 0],
        bounds[:, 1, 1],
        label="uncertain cell count",
        alpha=0.2,
        color="green",
    )
    ax[1].plot(counts[:, 1], color="green", label="predicted cell count")

    ax[0].legend()
    ax[1].legend()

    ax[1].set_xlabel("Frame")

    ax[0].set_ylabel("Cell count")
    ax[1].set_ylabel("Cell count")

    plt.tight_layout()

    fig.savefig(osp.join(basepath, "count_tunnel.png"))


if __name__ == "__main__":
    main()
