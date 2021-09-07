from acia.segm.output import VideoExporter
import numpy as np
import pandas as pd
import tqdm as tqdm
import cv2
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

from acia.segm.output import VideoExporter
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
from acia.base import ImageRoISource
import cv2
import numpy as np
from PIL import Image
import pandas as pd


image_id = 470

# your user credentials
credentials = dict(
    username='root',
    password='omero',
    serverUrl='ibt056',
)

# combine images and rois
irs = ImageRoISource(
    OmeroSequenceSource(image_id, **credentials, channels=[1,2,3], colorList=['FFFFFF', 'FF0000', '00FF00']),
    OmeroRoISource(image_id, **credentials)
)

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

# get indices of meaningful clusters
red_index = np.argmax(kmeans.cluster_centers_[:,0])
green_index = np.argmax(kmeans.cluster_centers_[:,1])

# prepare colors for clusters
color = np.array(['yellow'] * 3)
color[red_index] = 'red'
color[green_index] = 'green'

# append clustering result to dataframe
df['label'] = kmeans.labels_
df['color'] = color[kmeans.labels_]

with VideoExporter('cell_clustering.avi', framerate=3) as ve:
    print('Loading data from server...')
    for i, (image, overlay) in enumerate(tqdm.tqdm(irs)):
        # draw all cell countours with their respective cluster color
        pil_image = Image.fromarray(image, 'RGB')
        overlay.draw(pil_image, outlineColor = lambda r: df[df['id'] == r.id]['color'])

        # convert to raw image
        raw_image = np.asarray(pil_image)

        # convert to bgr (for opencv output)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        # add frame to video
        ve.write(raw_image)