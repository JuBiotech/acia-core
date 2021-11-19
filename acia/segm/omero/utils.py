def getImage(conn, imageId: int):
    return conn.getObject('Image', imageId)

def getDataset(conn, datasetId: int):
    return conn.getObject('Dataset', datasetId)

def getProject(conn, projectId: int):
    return conn.getObject('Project', projectId)

def list_projects(conn):
    """List projects in the current user group

    Args:
        conn ([type]): Current omero BlitzGateway connection

    Returns:
        [List]: List of project wrappers
    """
    return conn.getObjects('Project')

def list_image_ids_in_dataset(conn, datasetId: int):
    return [image.getId() for image in conn.getObjects('Image', opts={'dataset': datasetId})]

def list_images_in_dataset(conn, datasetId: int):
    return [image for image in conn.getObjects('Image', opts={'dataset': datasetId})]

def list_datasets_in_project(conn, projectId: int):
    return conn.getObjects('Dataset', opts={'project': projectId})

def list_images_in_project(conn, projectId: int):
    return [image for dataset in list_datasets_in_project(conn, projectId=projectId) for image in dataset.listChildren()]

def get_image_name(conn, imageId: int):
    return conn.getObject('Image', imageId).getName()

def get_project_name(conn, projectId: int):
    return conn.getObject('Project', projectId).getName()

def image_iterator(conn, object):
    if object.OMERO_CLASS == 'Image':
        yield object
    if object.OMERO_CLASS == 'Dataset':
        for image in list_images_in_dataset(conn, object.getId()):
            yield image
    if object.OMERO_CLASS == 'Project':
        for image in list_images_in_project(conn, object.getId()):
            yield image
