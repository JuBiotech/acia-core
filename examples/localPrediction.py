from acia.segm.processor.offline import OfflineModel
from acia.segm.local import LocalSequenceSource, RoiStorer
from exportVideo import renderVideo

if __name__ == '__main__':
    '''
        Simple example to use the apis
    '''
    # create local machine learning model
    model = OfflineModel('model_zoo/htc/htc_tuned.py', 'model_zoo/htc/latest.pth', half=True)
    # create local image data source
    source = LocalSequenceSource('input/nd074_crop01.nd2 - nd074_crop01.nd2 (series 7)-6.tif')

    # perform overlay prediction
    result = model.predict(source.slice(0,40), tiling=True)

    print('Render Video...')
    renderVideo(source.slice(0,40), result.timeIterator())

    # store in rois
    RoiStorer.store(result, 'rois.zip')