from typing import List
from omero.gateway import ImageWrapper, DatasetWrapper, ProjectWrapper, BlitzGateway
import omero

def getImage(conn: BlitzGateway, imageId: int) -> ImageWrapper:
    """Get omero image by id
        Note: only images in your current group are accessible

    Args:
        conn (BlitzGateway): current omero connection
        imageId (int): image id

    Returns:
        ImageWrapper: image object
    """
    return conn.getObject('Image', imageId)

def getDataset(conn: BlitzGateway, datasetId: int) -> DatasetWrapper:
    """Get omero dataset by id
        Note: only datasets in your current group are accessible

    Args:
        conn (BlitzGateway): active omero connection
        datasetId (int): dataset id

    Returns:
        DatasetWrapper: dataset object
    """
    return conn.getObject('Dataset', datasetId)

def getProject(conn: BlitzGateway, projectId: int) -> ProjectWrapper:
    """Get omero project by id
        Note: only projects in your current group are accessible

    Args:
        conn (BlitzGateway): active omero connection
        projectId (int): project id

    Returns:
        ProjectWrapper: project object
    """
    return conn.getObject('Project', projectId)

def list_projects(conn: BlitzGateway) -> List[ProjectWrapper]:
    """List projects in the current user group
        Note: only projects in your current group are accessible

    Args:
        conn (BlitzGateway): Current omero BlitzGateway connection

    Returns:
        List[ProjectWrapper]: List of project wrappers
    """
    return conn.getObjects('Project')

def list_image_ids_in_dataset(conn: BlitzGateway, datasetId: int) -> List[int]:
    """[summary]

    Args:
        conn (BlitzGateway): active omero connection
        datasetId (int): dataset id

    Returns:
        List[int]: array of all image ids of the dataset
    """
    return [image.getId() for image in conn.getObjects('Image', opts={'dataset': datasetId})]

def list_images_in_dataset(conn: BlitzGateway, datasetId: int) -> List[ImageWrapper]:
    """List all images in the omero dataset

    Args:
        conn (BlitzGateway): active omero connection
        datasetId (int): dataset id

    Returns:
        List[ImageWrapper]: List of omero images
    """
    return [image for image in conn.getObjects('Image', opts={'dataset': datasetId})]

def list_datasets_in_project(conn: BlitzGateway, projectId: int) -> List[DatasetWrapper]:
    return conn.getObjects('Dataset', opts={'project': projectId})

def list_images_in_project(conn: BlitzGateway, projectId: int) -> List[ImageWrapper]:
    return [image for dataset in list_datasets_in_project(conn, projectId=projectId) for image in dataset.listChildren()]

def get_image_name(conn: BlitzGateway, imageId: int) -> str:
    return conn.getObject('Image', imageId).getName()

def get_project_name(conn: BlitzGateway, projectId: int) -> str:
    return conn.getObject('Project', projectId).getName()

def image_iterator(conn: BlitzGateway, object) -> ImageWrapper:
    if object.OMERO_CLASS == 'Image':
        yield object
    if object.OMERO_CLASS == 'Dataset':
        for image in list_images_in_dataset(conn, object.getId()):
            yield image
    if object.OMERO_CLASS == 'Project':
        for image in list_images_in_project(conn, object.getId()):
            yield image


def create_project(conn: BlitzGateway, project_name: str)-> ProjectWrapper:
    new_project = ProjectWrapper(conn, omero.model.ProjectI())
    new_project.setName(project_name)
    new_project.save()
    return new_project

def create_dataset(conn: BlitzGateway, projectId: int, dataset_name: str)-> DatasetWrapper:
    # Use omero.gateway.DatasetWrapper:
    new_dataset = DatasetWrapper(conn, omero.model.DatasetI())
    new_dataset.setName(dataset_name)
    new_dataset.save()
    # Can get the underlying omero.model.DatasetI with:
    dataset_obj = new_dataset._obj

    # Create link to project
    link = omero.model.ProjectDatasetLinkI()
    # We can use a 'loaded' object, but we might get an Exception
    # link.setChild(dataset_obj)
    # Better to use an 'unloaded' object (loaded = False)
    link.setChild(omero.model.DatasetI(dataset_obj.id.val, False))
    link.setParent(omero.model.ProjectI(projectId, False))
    conn.getUpdateService().saveObject(link)

    return new_dataset