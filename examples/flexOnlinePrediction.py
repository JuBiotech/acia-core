from acia.segm.omero.storer import OmeroRoIStorer, OmeroSequenceSource
from acia.segm.processor.online import FlexibleOnlineModel, ModelDescriptor, OnlineModel
from acia.segm.filter import NMSFilter
from acia.segm.local import LocalSequenceSource, RoiStorer


if __name__ == '__main__':
    '''
        Simple example to use the apis
    '''

    imageId = 1351

    OmeroRoIStorer.clear(imageId, "bwollenhaupt", "bwollenhaupt", "ibt056")

    model_desc = ModelDescriptor(
        repo="https://gitlab+deploy-token-1:jzCPzEwRQacvqp8z2an9@jugit.fz-juelich.de/mlflow-executors/cellpose-executor.git",
        entry_point="main",
        version="main"
    )

    # connect to remote machine learning model
    model = FlexibleOnlineModel('http://ibt056/segService/batch-image-prediction/', model_desc, batch_size=10+1)

    # create local image data source
    source = OmeroSequenceSource(imageId, "bwollenhaupt", "bwollenhaupt", "ibt056", channels=[2])#, range=list(range(50)))
    #source = LocalSequenceSource('input/PHH2.nd2-PHH2.nd2(series6)_rois-1_70_final.tif')

    # perform overlay prediction
    print("Perform Prediction...")
    result = model.predict(source)

    # filter cell detections
    print("Filter detections")
    result = NMSFilter.filter(result, iou_thr=0.5, mode='i')

    # store detections in omero
    print("Save results...")
    OmeroRoIStorer.store(result, imageId, "bwollenhaupt", "bwollenhaupt", 'ibt056')
    #RoiStorer.store(result, 'rois.zip')