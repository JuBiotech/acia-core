from acia.segm.omero.storer import OmeroRoIStorer, OmeroSequenceSource
from acia.segm.omero.utils import get_image_name, get_project_name, list_images_in_project
from acia.segm.processor.online import OnlineModel
from acia.segm.filter import NMSFilter
from omero.gateway import BlitzGateway
import tqdm
import getpass

import logging

if __name__ == '__main__':
    '''
        Simple example to use the apis
    '''

    username = "root"
    serverUrl = "ibt056"
    password = getpass.getpass(f'Password for {username}@{serverUrl}: ')

    # connect to remote machine learning model
    model = OnlineModel('http://ibt056/pt/predictions/cellcmaskrcnn/')

    projectId = 151

    print("Connect to omero...")
    with BlitzGateway(username, password, host=serverUrl, port=4064, secure=True) as conn:
        print("Connection established!")
        projectName = get_project_name(conn, projectId)
        print(f"Scanning project'{projectName}'")
        image_list = list_images_in_project(conn, projectId)
        print(get_image_name(conn, 1))
        print(conn.getObject('Image', 1).getProject())

    #exit(1)

    print(image_list)
    print(len(image_list))
    #exit(1)

    for imageId in tqdm.tqdm(image_list):
        try:
            # create local image data source
            source = OmeroSequenceSource(imageId, username, password, serverUrl)
            #source = LocalSequenceSource('input/PHH2.nd2-PHH2.nd2(series6)_rois-1_70_final.tif')

            # perform overlay prediction
            projectName = source.projectName()
            datasetName = source.datasetName()
            imageName = source.imageName()
            print(f"Predict {projectName} > {datasetName} > {imageName}")
            result = model.predict(source, params={'test_cfg.rcnn.nms.iou_threshold': 0.8, 'test_cfg.rcnn.score_thr': 0.25})

            # filter cell detections
            print("Filter detections")
            result = NMSFilter.filter(result, iou_thr=0.2, mode='i')

            # store detections in omero
            print("Save results...")
            OmeroRoIStorer.store(result, imageId, username, password, serverUrl)
        except:
            logging.error(f"Error while processing {projectName} > {datasetName} > {imageName}")
        #RoiStorer.store(result, 'rois.zip')