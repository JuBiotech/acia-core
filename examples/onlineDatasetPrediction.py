from acia.segm.omero.storer import OmeroRoIStorer, OmeroSequenceSource
from acia.deploy import OnlineModel, LocalSequenceSource, RoiStorer, NMSFilter
from acia.segm.omero.utils import list_image_ids_in_dataset

from omero.gateway import BlitzGateway
import tqdm.auto as tqdm

if __name__ == '__main__':
    '''
        Simple example to use the apis
    '''

    username = "root"
    password = "omero"
    serverUrl = "ibtomero"

    # connect to remote machine learning model
    model = OnlineModel('http://ibt056/pt/predictions/cellcmaskrcnn/')

    datasetId = 201

    with BlitzGateway(username, password, host=serverUrl) as conn:
        image_list = list_image_ids_in_dataset(conn, datasetId)

    print(image_list)

    for imageId in tqdm.tqdm(image_list):
        # create local image data source
        source = OmeroSequenceSource(imageId, "root", "omero", "ibt056")
        #source = LocalSequenceSource('input/PHH2.nd2-PHH2.nd2(series6)_rois-1_70_final.tif')

        # perform overlay prediction
        print("Perform Prediction...")
        result = model.predict(source, params={'test_cfg.rcnn.nms.iou_threshold': 0.8, 'test_cfg.rcnn.score_thr': 0.25})

        # filter cell detections
        print("Filter detections")
        result = NMSFilter.filter(result, iou_thr=0.2, mode='i')

        # store detections in omero
        print("Save results...")
        OmeroRoIStorer.store(result, imageId, "root", "omero", 'ibt056')
        #RoiStorer.store(result, 'rois.zip')