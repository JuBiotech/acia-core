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

counts_green = []
counts_red = []

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

            counts_green.append(len(frame_df[frame_df['label'] == green_index]))
            counts_red.append(len(frame_df[frame_df['label'] == red_index]))

plt.close('all')

# plot absolute counts
plt.plot(counts_green, label='green cells', color='green')
plt.plot(counts_red, label='red cells', color='red')
plt.title('Absolute cell counts')
plt.xlabel('Frame')
plt.ylabel('Cell count')
plt.legend()
plt.savefig('cell_count.png')

total_count = np.array(counts_red) + np.array(counts_green)

counts_red_rel = np.array(counts_red) / total_count
counts_green_rel = np.array(counts_green) / total_count

plt.close('all')
plt.fill(counts_red_rel)
plt.fill_between(list(enumerate(counts_red_rel)), counts_red_rel, counts_green_rel + counts_red_rel)
plt.savefig('cell_count_rel.png')