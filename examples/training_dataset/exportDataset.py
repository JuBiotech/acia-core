""" Example to export segmentation data into COCO dataset format """

import getpass
import logging
import re
from typing import List

import omero
from omero.gateway import BlitzGateway

from acia.base import ImageRoISource
from acia.segm.omero.storer import OmeroRoISource, OmeroSequenceSource
from acia.segm.omero.utils import list_datasets_in_project
from acia.segm.output import CocoDataset


def has_all_tags(object, tag_list: List[str] = None, case_sensitive=False) -> bool:
    """[summary]

    Args:
        object ([type]): Omero object
        tag_list (List[str], optional): List of string tags that the object should have. Defaults to [].
        case_sensitive (bool, optional): Whether to match tag names with case sensitivity. Defaults to False.

    Returns:
        [bool]: Returns true if the object contains all tags. False otherwise
    """

    if tag_list is None:
        tag_list = []

    def str_view(input: str) -> str:
        if case_sensitive:
            return input
        else:
            return input.lower()

    tag_list = tag_list.copy()

    tag_list = list(map(str_view, tag_list))

    for ann in object.listAnnotations():
        if ann.OMERO_TYPE == omero.model.TagAnnotationI:
            if str_view(ann.getTextValue()) in tag_list:
                # remove the tag from the list (successful match)
                del tag_list[tag_list.index(str_view(ann.getTextValue()))]

        if len(tag_list) == 0:
            # if all tags have been matched stop
            return True

    # There are still missing tags
    return False


def goldStandardInfo(image):
    annotations = image.listAnnotations()
    for ann in annotations:
        if ann.OMERO_TYPE != omero.model.MapAnnotationI:
            continue
        key, value = ann.getValue()[0]
        if key != "gold-standard":
            continue

        res = re.match(r"(?P<start>\d+):(?P<end>\d+)", value)

        if res:
            data = res.groupdict()

            start = int(data["start"])
            end = int(data["end"])

            # sequence including start end end (start count with 1)
            return range(start - 1, end)

    return []


def irs_from_image_list(image_list, conn):
    irs_list = []
    for image_id in image_list:
        image = conn.getObject("Image", image_id)
        gold_range = list(goldStandardInfo(image))
        if len(gold_range) == 0:
            logging.warning("%s: No gold-standard range defined!", image.getName())
            # continue
        irs = ImageRoISource(
            OmeroSequenceSource(imageId=image_id, **omero_cred, range=gold_range),
            OmeroRoISource(image_id, **omero_cred, range=gold_range),
        )
        irs_list.append(irs)

    return irs_list


if __name__ == "__main__":
    serverUrl = "ibt056"
    username = "root"
    password = getpass.getpass(f"Password for {username}@{serverUrl}: ")

    omero_cred = {"username": username, "serverUrl": serverUrl, "password": password}

    selected_projects = [101]

    logging.info("Connect to omero...")
    with BlitzGateway(
        username, password, host=serverUrl, port=4064, secure=True
    ) as conn:
        tags = conn.getObjects("TagAnnotation")

        print("Available tags:")
        for tag in tags:
            print("\t", tag.textValue)

        print("Datasets with 'C. glutamicum': ")
        dataset_list = []
        if selected_projects is None:
            # search all projects
            for dataset in conn.getObjects("Dataset"):
                if has_all_tags(dataset, ["C. glutamicum"]):
                    print("\t", dataset.getName())
                    dataset_list.append(dataset.getId())
        else:
            # search only selected projects
            for project_id in selected_projects:
                for dataset in list_datasets_in_project(
                    conn=conn, projectId=project_id
                ):
                    dataset_list.append(dataset.getId())

        print("Ground truth data for C. glutamicum")
        image_list = []

        image_train_list = []
        image_val_list = []

        for dataset_id in dataset_list:
            dataset = conn.getObject("Dataset", dataset_id)
            all_gold_standard = False
            all_train = False
            all_val = False
            if has_all_tags(dataset, ["gold-standard"]):
                all_gold_standard = True
            if has_all_tags(dataset, ["train"]):
                all_train = True
            if has_all_tags(dataset, ["val"]):
                all_val = True

            for image in conn.getObjects("Image", opts={"dataset": dataset_id}):
                if all_gold_standard or has_all_tags(image, ["gold-standard"]):
                    dataset = image.getParent()
                    project = dataset.getParent()
                    print(
                        f"{project.getName()} > {dataset.getName()} > {image.getName()}"
                    )

                    if all_train or has_all_tags(image, ["train"]):
                        image_train_list.append(image.getId())
                    elif all_val or has_all_tags(image, ["val"]):
                        image_val_list.append(image.getId())
                    else:
                        logging.warning(
                            "No dataset (train | val) association: put into train"
                        )
                        image_train_list.append(image.getId())

                    # add the image to the list
                    # image_list.append(image.getId())

        # combine all the image sequences with their rois
        irs_list = []

        # TODO: Validation and test dataset should be relatively fixed (at least test)

        train_irs_list = irs_from_image_list(image_train_list, conn)
        val_irs_list = irs_from_image_list(image_val_list, conn)

        print("Training: " + str(image_train_list))
        print("Validation: " + str(image_val_list))

        print("Downloading Training dataset...")
        cd = CocoDataset()
        cd.add(train_irs_list)
        cd.write("coco")

        print("Downloading Validation dataset...")
        cd = CocoDataset()
        cd.add(val_irs_list)
        cd.write("coco", "val")
