from acia.deploy import OfflineModel, LocalSequenceSource, RoiStorer

if __name__ == '__main__':
    '''
        Simple example to use the apis
    '''
    # create local machine learning model
    model = OfflineModel('train_work_dirs/train_data/me_htc_2_False_True_15_True_12/me_htc_2_False_True_15_True_12.py', 'train_work_dirs/train_data/me_htc_2_False_True_15_True_7/latest.pth', half=True)
    # create local image data source
    source = LocalSequenceSource('input/PHH2.nd2-PHH2.nd2(series6)_rois-1_70_final.tif')

    # perform overlay prediction
    result = model.predict(source)

    # store in rois
    RoiStorer.store(result, 'rois.zip')