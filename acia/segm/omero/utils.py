def list_image_ids_in_dataset(conn, datasetId: int):
    return [image.getId() for image in conn.getObjects('Image', opts={'dataset': datasetId})]

def list_images_in_dataset(conn, datasetId: int):
    return [image for image in conn.getObjects('Image', opts={'dataset': datasetId})]

def list_images_in_project(conn, projectId: int):
    return [image for dataset in list_datasets_in_project(conn, projectId=projectId) for image in dataset.listChildren()]

def get_image_name(conn, imageId: int):
    return conn.getObject('Image', imageId).getName()

def get_project_name(conn, projectId: int):
    return conn.getObject('Project', projectId).getName()