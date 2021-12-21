from scipy.sparse import base
from acia.segm.output import VideoExporter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tqdm as tqdm
import cv2
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import os.path as osp
from config import basepath

datapath = osp.join(basepath, 'datapoints.pkl')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

def main():

    # read dataset
    df = pd.read_pickle(datapath)

    # perform standardization
    std_slc = StandardScaler()
    transform = std_slc.fit(df[['red', 'green']])

    # transform data
    X_std = transform.transform(df[['red', 'green']])

    # execute kmeans with 3 clusters on transformed data
    cluster = KMeans(3)
    kmeans = cluster.fit(X_std)

    df['label'] = kmeans.labels_

    # get indices of meaningful clusters
    red_index = np.argmax(kmeans.cluster_centers_[:,0])
    green_index = np.argmax(kmeans.cluster_centers_[:,1])

    # prepare colors for clusters
    color = np.array(['yellow'] * 3)
    color[red_index] = 'red'
    color[green_index] = 'green'

    df['color'] = color[kmeans.labels_]

    # store pickle with color information
    df.to_pickle(datapath)

    #print('Clustering...')
    #clustering = DBSCAN(eps=0.1, min_samples=2).fit(df[['red', 'green']])
    #print('Done')

    # get the cluster centers
    inv_centroids = transform.inverse_transform(kmeans.cluster_centers_)
    centroids = inv_centroids

    counts_green = []
    counts_red = []

    with VideoExporter(osp.join(basepath, 'datapoints.avi'), 3) as ve:
        with VideoExporter(osp.join(basepath, 'clusters.avi'), 3) as veC:
            for frame in tqdm.tqdm(range(df['frame'].max())):
                frame_df = df[df['frame'] == frame]

                # perform label predictions on normalized data
                prediction = kmeans.predict(transform.transform(frame_df[['red', 'green']]))

                # scatter data with correct colors
                plt.scatter(frame_df['red'], frame_df['green'], c= color[prediction], s=50, alpha=0.5)
                # scatter cluster centers
                plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=50, marker='+')
                # make figure
                plt.xlim((0.2, 0.7))
                plt.ylim((0.2, 0.7))
                plt.title('Frame: %03d' % frame)
                plt.xlabel('red')
                plt.ylabel('green')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(osp.join(basepath, 'cluster.png'),  dpi=300)
                # read figure from file
                img = cv2.imread(osp.join(basepath, 'cluster.png'))
                # write figure into video
                veC.write(image=img)
                plt.close('all')


                fig, ax = plt.subplots()
                sns.displot(frame_df, x='red', y='green')
                plt.title('Frame: %03d' % frame)
                plt.tight_layout()
                plt.savefig(osp.join(basepath, 'datapoints.png'))
                plt.close('all')

                img = cv2.imread(osp.join(basepath, 'datapoints.png'))

                ve.write(image=img)

                counts_green.append(len(frame_df[frame_df['label'] == green_index]))
                counts_red.append(len(frame_df[frame_df['label'] == red_index]))

    plt.close('all')

    fig, ax1 = plt.subplots()
    # plot absolute counts
    plt.plot(counts_green, label='Green cells', color='green')
    plt.plot(counts_red, label='Red cells', color='red')
    plt.title('Cell Counts')
    plt.xlabel('Frame')
    plt.ylabel('Absolute Cell Count')
    plt.xlim((0, len(counts_green)))

    ax1.grid(True)

    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    #color = 'tab:blue'
    #ax2.set_ylabel(r'Ratio $\frac{red}{red+green}$', color=color)  # we already handled the x-label with ax1
    #ratio = np.array(counts_red) / (np.array(counts_green) + np.array(counts_red))
    #trunc_ratio = ratio[30:]
    #ax2.plot(np.array(range(len(trunc_ratio)))+30, trunc_ratio, color=color, label='Ratio')
    #ax2.tick_params(axis='y', labelcolor=color)


    plt.legend()
    plt.savefig(osp.join(basepath, 'cell_count.png'), dpi=300)

    # compute rel counts
    '''total_count = np.array(counts_red) + np.array(counts_green)

    counts_red_rel = np.array(counts_red) / total_count
    counts_green_rel = np.array(counts_green) / total_count

    plt.close('all')
    plt.fill(counts_red_rel)
    plt.fill_between(list(enumerate(counts_red_rel)), counts_red_rel, counts_green_rel + counts_red_rel)
    plt.savefig(osp.join(basepath, 'cell_count_rel.png'))'''

if __name__ == '__main__':
    main()