def list_images(conn, datasetId: int):
    return [image.getId() for image in conn.getObjects('Image', opts={'dataset': datasetId})]