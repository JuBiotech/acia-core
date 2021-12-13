from scipy.sparse import base
from acia.segm.output import VideoExporter
from examples.analysis.fluorescence.config import clustering
import numpy as np
from config import basepath
import os.path as osp
import matplotlib.pyplot as plt
import cv2
import tqdm

def compute_area_bounds(data, color):
    areas = data[data['color'] == color]['area']
    if len(areas):
        return np.percentile(areas, 5), np.percentile(areas, 95)
    else:
        return None

def area_distribution(data):
    areas = data['area']
    if len(areas) > 0:
        lower = np.percentile(areas, 5)
        upper = np.percentile(areas, 95)
    else:
        lower = np.nan
        upper = np.nan
    
    return np.mean(data['area']), np.std(data['area']), lower, upper

if __name__ == '__main__':

    # get the clustering in red and green
    df, transform, kmeans, red_index, green_index = clustering()

    # determine the individual frames
    frames = np.unique(df['frame'])

    with VideoExporter(osp.join(basepath, 'area_distributions.avi'), 3) as ve:
        for frame in tqdm.tqdm(frames):
            # get frame data frame
            frame_df = df[df['frame'] == frame]

            # compute red/green area distributions
            red_area_dist = area_distribution(frame_df[frame_df['color'] == 'red'])
            green_area_dist = area_distribution(frame_df[frame_df['color'] == 'green'])

            # log them
            print(red_area_dist, green_area_dist)

            # create 2 plots
            figs, ax = plt.subplots(2, 1)

            # plot both area distributions at the current frame
            ax[0].hist(frame_df[frame_df['color'] == 'red']['area'], color='red', alpha=0.5)
            ax[1].hist(frame_df[frame_df['color'] == 'green']['area'], color='green', alpha=0.5)

            # keep the xlimits the same
            ax[0].set_xlim((0, 500))
            ax[1].set_xlim((0, 500))

            # add to video
            plt.savefig(osp.join(basepath, 'area_dist.png'))
            plt.close('all')
            img = cv2.imread(osp.join(basepath, 'area_dist.png'))
            ve.write(image=img)

