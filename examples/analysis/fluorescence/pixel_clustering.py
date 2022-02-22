from examples.analysis.fluorescence.config import clustering
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
from acia.base import ImageRoISource
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import basepath
import os.path as osp
import tqdm.auto as tqdm

def cluster_pixels(image, transform, kmeans, kernel_size=9):
    kernel = np.ones((kernel_size, kernel_size),np.float32)/(kernel_size*kernel_size)
    dst = cv2.filter2D(image,-1,kernel)

    norm_image = image.astype(float) / 255

    individual_pixels = np.reshape(norm_image[:,:,:2], (-1,2))

    transformed_pixels = transform.transform(individual_pixels)

    predicted_labels = kmeans.predict(transformed_pixels)   

    return predicted_labels 

def main():

    # omero id of the image (Image ID)
    image_id = 470
    # fluorescence channels you want to monitor (usually 1 is the phase contrast)
    fluorescence_channels = [2, 3]

    # your user credentials for omero
    credentials = dict(
        username='root',
        password='omero',
        serverUrl='ibt056',
    )

    # combine images and rois
    #irs = ImageRoISource(
    oss = OmeroSequenceSource(image_id, **credentials, channels=fluorescence_channels, colorList=['FF0000', '00FF00'])
    #    OmeroRoISource(image_id, **credentials)
    #)

    df, transform, kmeans, red_index, green_index = clustering()

    kernel_size = 3

    counts = []

    for image in tqdm.tqdm(oss):
        # predict cluster labels for individual pixels
        predicted_labels = cluster_pixels(image, transform, kmeans, kernel_size)

        red_count = np.sum(predicted_labels == red_index)
        green_count = np.sum(predicted_labels == green_index)

        print(red_count, green_count)

        counts.append((red_count, green_count))

    counts = np.array(counts)

    plt.plot(counts[:,0], label="red pixel count", c='red')
    plt.plot(counts[:,1], label="green pixel count", c='green')
    plt.title('Absolute pixel counts')
    plt.xlabel('Frame')
    plt.ylabel('Pixel count')
    plt.tight_layout()
    plt.legend()

    plt.savefig(osp.join(basepath, 'pixel_counts.png'))






if __name__ == '__main__':
    main()