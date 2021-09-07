from acia.segm.output import VideoExporter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tqdm as tqdm
import cv2
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# read dataset
df = pd.read_pickle('datapoints.pkl')

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

df.to_pickle('datapoints.pkl')

#print('Clustering...')
#clustering = DBSCAN(eps=0.1, min_samples=2).fit(df[['red', 'green']])
#print('Done')

# get the cluster centers
inv_centroids = transform.inverse_transform(kmeans.cluster_centers_)
centroids = inv_centroids


with VideoExporter('datapoints.avi', 3) as ve:
    with VideoExporter('clusters.avi', 3) as veC:
        for frame in tqdm.tqdm(range(df['frame'].max())):
            frame_df = df[df['frame'] == frame]

            # perform label predictions on normalized data
            prediction = kmeans.predict(transform.transform(frame_df[['red', 'green']]))

            # scatter data with correct colors
            plt.scatter(frame_df['red'], frame_df['green'], c= color[prediction], s=50, alpha=0.5)
            # scatter cluster centers
            plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=50, marker='+')
            # make figure
            plt.title('Frame: %03d' % frame)
            plt.xlabel('red')
            plt.ylabel('green')
            plt.tight_layout()
            plt.savefig('cluster.png')
            # read figure from file
            img = cv2.imread('cluster.png')
            # write figure into video
            veC.write(image=img)
            plt.close('all')


            fig, ax = plt.subplots()
            sns.displot(frame_df, x='red', y='green')
            plt.title('Frame: %03d' % frame)
            plt.tight_layout()
            plt.savefig('datapoints.png')
            plt.close('all')

            img = cv2.imread('datapoints.png')

            ve.write(image=img)
