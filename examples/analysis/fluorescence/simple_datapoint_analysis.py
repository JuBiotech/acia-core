""" Example to classify fluorescence datapoints into different clusters"""

import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm.auto as tqdm
from config import basepath  # , datapoints_path

from acia.segm.output import VideoExporter

datapoints_path = osp.join(basepath, "datapoints_labeled.pkl")


# TODO: make function shorter
# pylint: disable=R0915
def main():

    cluster_colors = np.array(["yellow", "red", "green"])

    red_index = 1
    green_index = 2

    green_threshold = 0.3
    max_area = 750

    # read dataset
    df = pd.read_pickle(datapoints_path)

    areas = df["area"]

    labels = np.zeros(len(df), dtype=np.int32)

    labels[df["green"] >= green_threshold] = 2

    labels[labels == 0] = 1

    labels[areas >= max_area] = 0

    df["label"] = labels
    df["color"] = cluster_colors[labels]

    # store pickle with color information
    df.to_pickle(datapoints_path)

    counts_green = []
    counts_red = []

    with VideoExporter(osp.join(basepath, "datapoints.avi"), 3) as ve:
        with VideoExporter(osp.join(basepath, "clusters.avi"), 3) as veC:
            for frame in tqdm.tqdm(range(df["frame"].max())):
                frame_df = df[df["frame"] == frame]
                # perform label predictions on normalized data
                # prediction = kmeans.predict(transform.transform(frame_df[['red', 'green']]))

                # scatter data with correct colors
                plt.scatter(
                    frame_df["red"],
                    frame_df["green"],
                    c=frame_df["color"],
                    s=50,
                    alpha=0.5,
                )
                # make figure
                plt.xlim((0, 0.7))
                plt.ylim((0, 0.7))
                plt.title(f"Frame: {frame:03d}")
                plt.xlabel("red")
                plt.ylabel("green")
                plt.tight_layout()
                plt.savefig(osp.join(basepath, "cluster.png"))
                # read figure from file
                img = cv2.imread(osp.join(basepath, "cluster.png"))
                # write figure into video
                veC.write(image=img)
                plt.close("all")

                plt.subplots()
                sns.displot(frame_df, x="red", y="green")
                plt.title(f"Frame: {frame:03d}")
                plt.tight_layout()
                plt.savefig(osp.join(basepath, "datapoints.png"))
                plt.close("all")

                img = cv2.imread(osp.join(basepath, "datapoints.png"))

                ve.write(image=img)

                counts_green.append(len(frame_df[frame_df["label"] == green_index]))
                counts_red.append(len(frame_df[frame_df["label"] == red_index]))

    plt.close("all")

    # plot absolute counts
    plt.plot(counts_green, label="green cells", color="green")
    plt.plot(counts_red, label="red cells", color="red")
    plt.title("Absolute cell counts")
    plt.xlabel("Frame")
    plt.ylabel("Cell count")
    plt.legend()
    plt.savefig(osp.join(basepath, "cell_count.png"))

    df = pd.DataFrame(
        np.array([counts_red, counts_green]).T, columns=["count red", "count green"]
    )
    df.to_csv(osp.join(basepath, "counts.csv"))


if __name__ == "__main__":
    main()
