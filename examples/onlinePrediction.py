from acia.segm.omero.storer import OmeroRoIStorer, OmeroSequenceSource
from acia.segm.processor.online import OnlineModel
from acia.segm.filter import NMSFilter
from acia.segm.local import LocalSequenceSource, RoiStorer


if __name__ == '__main__':
    '''
        Simple example to use the apis
    '''
    # connect to remote machine learning model
    model = OnlineModel('http://ibt056/pt/predictions/cellcmaskrcnn/')

    # create local image data source
    source = OmeroSequenceSource(2, "root", "omero", "ibt056")
    #source = LocalSequenceSource('input/PHH2.nd2-PHH2.nd2(series6)_rois-1_70_final.tif')

    # perform overlay prediction
    print("Perform Prediction...")
    result = model.predict(source, params={'test_cfg.rcnn.nms.iou_threshold': 0.8, 'test_cfg.rcnn.score_thr': 0.25})

    # filter cell detections
    print("Filter detections")
    result = NMSFilter.filter(result, iou_thr=0.5, mode='i')

    # store detections in omero
    print("Save results...")
    OmeroRoIStorer.store(result, 2, "root", "omero", 'ibt056')
    #RoiStorer.store(result, 'rois.zip')