"""Module for general visualization functionality
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import cv2
import moviepy.editor as mpy
import networkx as nx
import numpy as np
import pint
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from acia import ureg
from acia.base import BaseImage, ImageSequenceSource, Overlay
from acia.segm.local import InMemorySequenceSource, LocalImage

from .utils import strfdelta

# loda the deja vu sans default font
default_font = font_manager.findfont("DejaVu Sans")


def draw_scale_bar(
    image_iterator,
    xy_position: tuple[int, int],
    size_of_pixel,
    bar_width,
    bar_height,
    color=(255, 255, 255),
    font_size=25,
    font_path=default_font,
    background_color=None,
    background_margin_pixel=3,
):
    """Draws a scale bar on all images of an image sequence or iterable image array

    Args:
        image_iterator: image sequence or iterator over images
        xy_position (tuple[int, int]): lower left xy position of the scale bar
        size_of_pixel (_type_): metric size of a pixel (e.g. 0.007 * ureg.micrometer)
        bar_width (_type_): width of the scale bar (e.g. 5 * ureg.micrometer)
        short_title (str, optional): Short title of the unit to be displayed. Defaults to "μm".
        color (tuple, optional): Color of scale bar and text. Defaults to (255, 255, 255).
        font_size (int, optional): text font size. Defaults to 25.
        font_path (str, optional): text font. Defaults to "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf".
        background_color: color for a potential background rectangle (e.g. (0, 0, 0)). Defaults to None (no background drawn).
        background_margin_pixel: pixels of margin for the background rectangle

    Yields:
        np.ndarray | LocalImage: Image in numpy format or LocalImage (depending on the input format)
    """

    # create pint quantities (values and units)
    bar_width = ureg.Quantity(bar_width)
    bar_height = ureg.Quantity(bar_height)
    size_of_pixel = ureg.Quantity(size_of_pixel)

    # load font
    font = ImageFont.truetype(font_path, font_size)

    # compute width and height of the scale bar in pixels (we need to round here)
    bar_pixel_width = int(
        np.round((bar_width / size_of_pixel).to_base_units().magnitude)
    )
    bar_pixel_height = int(
        np.round((bar_height / size_of_pixel).to_base_units().magnitude)
    )

    # extract position
    xstart, ystart = xy_position

    for image in image_iterator:

        # do we have a wrapped image?
        is_wrapped = isinstance(image, BaseImage)

        # unwrap if necessary
        if is_wrapped:
            image = image.raw

        # compute text size
        text = f"{bar_width:~P}"
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # get size of text
        left, top, right, bottom = draw.textbbox((xstart, ystart), text, font=font)

        text_width = right - left
        text_height = bottom - top

        if background_color:
            cv2.rectangle(
                image,
                (xstart - background_margin_pixel, ystart + background_margin_pixel),
                (
                    xstart + bar_pixel_width + background_margin_pixel,
                    ystart
                    - text_height
                    - bar_pixel_height
                    - 5
                    - background_margin_pixel,
                ),
                background_color,
                -1,
            )

        # draw scale bar
        cv2.rectangle(
            image,
            (xstart, ystart),
            (xstart + bar_pixel_width, ystart - bar_pixel_height),
            (255, 255, 255),
            -1,
        )

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # draw text centered and with distance to the scale bar
        draw.text(
            (
                xstart + bar_pixel_width / 2 - text_width / 2,
                ystart - text_height - bar_pixel_height - 10,
            ),
            text,
            fill=color,
            font=font,
        )

        # convert PIL image back to numpy
        image = np.array(img_pil)

        # do the image wrapping
        if is_wrapped:
            yield LocalImage(image)
        else:
            yield image


def draw_time(
    image_iterator,
    xy_position,
    time_step,
    color=(255, 255, 255),
    font_size=25,
    font_path=default_font,
    background_color=None,
    background_margin_pixel=3,
):
    """Draw time onto images

    Args:
        image_iterator (_type_): image sequence or iterator over images
        xy_position (tuple[int, int]): lower left xy position of the time text
        time_step (_type_): time step between images (e.g. 15 * ureg.minute or "15 minute")
        color (_type): Color of the time text. Defaults to (255, 255, 255) which is white.
        font_size (int, optional): text font size. Defaults to 25.
        font_path (str, optional): text font. Defaults to "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf".
        background_color: color for a potential background rectangle (e.g. (0, 0, 0)). Defaults to None (no background drawn).
        background_margin_pixel: pixels of margin for the background rectangle

    Yields:
        _type_: _description_
    """

    time_step = ureg.Quantity(time_step)

    # load font
    font = ImageFont.truetype(font_path, font_size)

    for frame, image in enumerate(image_iterator):

        # do we have a wrapped image?
        is_wrapped = isinstance(image, BaseImage)

        # unwrap if necessary
        if is_wrapped:
            image = image.raw

        # convert to pillow image
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # extract time in hours and minutes
        time = (frame * time_step).to(ureg.hour)
        hours = int(np.floor(time.magnitude))
        minutes = int(np.round((time - hours * ureg.hour).to("minute").magnitude))

        time_text = f"Time: {hours:2d}:{minutes:02d} h"

        if background_color:
            # get size of text
            left, top, right, bottom = draw.textbbox(xy_position, time_text, font=font)

            text_width = right - left
            text_height = bottom - top

            x, y = xy_position

            cv2.rectangle(
                image,
                (x - background_margin_pixel, y - background_margin_pixel),
                (
                    x + text_width + background_margin_pixel,
                    y + text_height + background_margin_pixel + 5,
                ),
                background_color,
                -1,
            )

            # convert to pillow image
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

        # draw on image
        draw.text(xy_position, time_text, fill=color, font=font)

        # convert PIL image back to numpy
        image = np.array(pil_image)

        # do the image wrapping
        if is_wrapped:
            yield LocalImage(image)
        else:
            yield image


class VideoExporter:
    """
    Wrapper for opencv video writer. Simplifies usage
    """

    def __init__(self, filename, framerate, codec="MJPG"):
        self.filename = filename
        self.framerate = framerate
        self.out = None
        self.frame_height = None
        self.frame_width = None
        self.codec = codec

    def __del__(self):
        if self.out:
            self.close()

    def write(self, image):
        height, width = image.shape[:2]
        if self.out is None:
            self.frame_height, self.frame_width = image.shape[:2]
            self.out = cv2.VideoWriter(
                self.filename,
                cv2.VideoWriter_fourcc(*self.codec),
                self.framerate,
                (self.frame_width, self.frame_height),
            )
        if self.frame_height != height or self.frame_width != width:
            logging.warning(
                "You add images of different resolution to the VideoExporter. This may cause problems (e.g. black video output)!"
            )
        self.out.write(image)

    def close(self):
        if self.out:
            self.out.release()
            self.out = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.out is None:
            logging.warning(
                "Closing video writer without any images written and no video output generated! Did you forget to write the images="
            )
        self.close()


class VideoExporter2:
    """
    Wrapper for opencv video writer. Simplifies usage
    """

    def __init__(
        self, filename: Path, framerate: int, codec="mjpeg", ffmpeg_params=None
    ):
        self.filename = Path(filename)
        self.framerate = framerate
        self.codec = codec

        if ffmpeg_params is None:
            ffmpeg_params = []

        self.ffmpeg_params = ffmpeg_params

        self.images = []

    @staticmethod
    def default_vp9(
        filename: Path,
        framerate: int,
    ):
        ffmpeg_params = ["-crf", "30", "-b:v", "0", "-speed", "1"]
        return VideoExporter2(
            filename, framerate, codec="libvpx-vp9", ffmpeg_params=ffmpeg_params
        )

    @staticmethod
    def fast_vp9(
        filename: Path,
        framerate: int,
    ):
        ffmpeg_params = ["-crf", "35", "-b:v", "0", "-speed", "3"]
        return VideoExporter2(
            filename, framerate, codec="libvpx-vp9", ffmpeg_params=ffmpeg_params
        )

    @staticmethod
    def default_h264(
        filename: Path,
        framerate: int,
    ):
        ffmpeg_params = ["-crf", "30", "-preset", "fast"]
        return VideoExporter2(
            filename, framerate, codec="libx264", ffmpeg_params=ffmpeg_params
        )

    @staticmethod
    def default_h265(filename: Path, framerate: int):
        ffmpeg_params = ["-crf", "26", "-preset", "fast"]
        return VideoExporter2(
            filename, framerate, codec="libx265", ffmpeg_params=ffmpeg_params
        )

    @staticmethod
    def default_mjpg(filename: Path, framerate: int):
        ffmpeg_params = []
        return VideoExporter2(
            filename, framerate, codec="mjpeg", ffmpeg_params=ffmpeg_params
        )

    # av1 not yet supported
    #    @staticmethod
    #    def default_av1(filename: Path, framerate: int, ffmpeg_params=["-crf", "26", "-preset", "2", "-strict", "2"]):
    #        return VideoExporter2(filename, framerate, codec="libaom-av1", ffmpeg_params=ffmpeg_params)

    def write(self, image):
        self.images.append(image)

    def close(self):
        if len(self.images) == 0:
            logging.warning(
                "Closing video writer without any images written and no video output generated! Did you forget to write the images?"
            )
        else:
            # do the video rendering
            clip = mpy.ImageSequenceClip(
                list(self.images),
                fps=self.framerate,
            )
            clip.write_videofile(
                str(self.filename.absolute()),
                codec=self.codec,
                ffmpeg_params=self.ffmpeg_params,
                # verbose=False,
                # logger=None,
            )
            self.images = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def render_segmentation(
    imageSource: ImageSequenceSource,
    overlay: Overlay,
    cell_color=(255, 255, 0),
) -> ImageSequenceSource:
    """Render a video of the time-lapse including the segmentaiton information.

    Args:
        imageSource (ImageSequenceSource): Your time-lapse source object.
        Overlay ([type]): Your source of RoIs for the image (e.g. cells).
        cell_color: rgb color of the cell outlines
    """

    if overlay is None:
        # when we have no rois -> create iterator that always returns None
        def always_none():
            while True:
                yield None

        overlay = iter(always_none())

    images = []

    for image, frame_overlay in tqdm(
        zip(imageSource, overlay.timeIterator()), desc="Render cell segmentation..."
    ):
        # extract the numpy image
        if isinstance(image, BaseImage):
            image = image.raw
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise Exception("Unsupported image type!")

        # copy image as we draw onto it
        image = np.copy(image)

        # Draw overlay
        if frame_overlay:
            image = frame_overlay.draw(image, cell_color)  # RGB format

        images.append(image)

    # return as sequence source again
    return InMemorySequenceSource(np.stack(images))


def render_cell_centers(
    image_source: ImageSequenceSource | np.ndarray,
    overlay: Overlay,
    center_color=(255, 255, 0),
    center_size=3,
) -> ImageSequenceSource:
    """Render a image sequence of the time-lapse with the cell centers.

    Args:
        imageSource (ImageSequenceSource): Your time-lapse source object.
        overlay (Overlay, optional): Your source of RoIs for the image (e.g. cells).
        center_color (tuple, optional): RGB color of the cell center circle. Defaults to (255, 255, 0).
        center_size (int, optional): Radius of the cell center circle (in pixels). Defaults to 3.

    Raises:
        ValueError: If we recognize unsupported image type or format

    Returns:
        ImageSequenceSource: The rendered image sequence
    """

    if overlay is None:
        # when we have no rois -> create iterator that always returns None
        def always_none():
            while True:
                yield None

        overlay = iter(always_none())

    images = []

    for image, frame_overlay in tqdm(
        zip(image_source, overlay.timeIterator()), desc="Render cell centers..."
    ):
        # extract the numpy image
        if isinstance(image, BaseImage):
            image = image.raw
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError("Unsupported image type!")

        # copy image as we draw onto it
        image = np.copy(image)

        # Draw overlay
        if frame_overlay:

            # compute all centers
            centers = [cont.center for cont in frame_overlay]

            for center in centers:
                int_center = tuple(map(int, center))

                cv2.circle(image, int_center, center_size, center_color, -1)

        images.append(image)

    image_stack = np.stack(images)

    if isinstance(ImageSequenceSource, np.ndarray):
        # return as raw numpy stack
        return image_stack
    else:
        # return as sequence source again
        return InMemorySequenceSource(image_stack)


def render_tracking(
    image_source: ImageSequenceSource,
    overlay: Overlay,
    tracking_graph: nx.DiGraph,
) -> ImageSequenceSource:
    """Render the tracking to an image source

    Args:
        image_source (ImageSequenceSource): Image source
        overlay (Overlay): overla of cell detections (for center points)
        tracking_graph (nx.DiGraph): the tracking graph where every cell detection is a node in the graph.

    Returns:
        ImageSequenceSource: Rendered image source
    """

    images = []

    contour_lookup = {cont.id: cont for cont in overlay}

    for image, frame_overlay in tqdm(
        zip(image_source, overlay.timeIterator()), desc="Render cell tracking..."
    ):

        np_image = np.copy(image.raw)

        for cont in frame_overlay:
            if cont.id in tracking_graph.nodes:
                edges = tracking_graph.out_edges(cont.id)

                born = tracking_graph.in_degree(cont.id) == 0

                for edge in edges:
                    source = contour_lookup[edge[0]].center
                    target = contour_lookup[edge[1]].center

                    line_color = (255, 0, 0)  # rgb: red

                    if len(edges) > 1:
                        line_color = (0, 0, 255)  # bgr: blue

                    cv2.line(
                        np_image,
                        tuple(map(int, source)),
                        tuple(map(int, target)),
                        line_color,
                        thickness=3,
                    )

                    if born:
                        cv2.circle(
                            np_image,
                            tuple(map(int, source)),
                            3,
                            (203, 192, 255),
                            thickness=1,
                        )

                if len(edges) == 0:
                    cv2.rectangle(
                        np_image,
                        np.array(cont.center).astype(np.int32) - 2,
                        np.array(cont.center).astype(np.int32) + 2,
                        (203, 192, 255),
                    )

        images.append(np_image)

    return InMemorySequenceSource(images)


def render_video(
    image_source: ImageSequenceSource,
    filename: str,
    framerate: int,
    codec: str,
    ffmpeg_params: list[str] = None,
) -> None:
    """Render video

    Args:
        image_source (ImageSequenceSource): sequence of images
        filename (str): video filename
        framerate (int): framerate of the video
        codec (str): the codec for video encoding
    """

    with VideoExporter2(
        filename, framerate=framerate, codec=codec, ffmpeg_params=ffmpeg_params
    ) as ve:
        for im in tqdm(image_source, desc="Encoding video..."):
            ve.write(im.raw)


def render_scalebar(
    image_source: Overlay,
    xy_position: tuple[int, int],
    size_of_pixel: pint.Quantity,
    bar_width: pint.Quantity,
    bar_height: pint.Quantity,
    color=(255, 255, 255),
    font_size=25,
    font_path=default_font,
    background_color: tuple[int, int, int] = None,
    background_margin_pixel=3,
    show_text=True,
) -> ImageSequenceSource:
    """Draws a scale bar on all images of an image sequence or iterable image array

    Args:
        image_source (Overlay): image sequence or iterator over images
        xy_position (tuple[int, int]): lower left xy position of the scale bar
        size_of_pixel (pint.Quantity): metric size of a pixel (e.g. 0.007 * ureg.micrometer)
        bar_width (pint.Quantity): width of the scalebar (e.g. 5 * ureg.micrometer). Also the text over the bar.
        bar_height (pint.Quantity): height of the scalebar.
        color (tuple, optional): Color of the scalebar and text. Defaults to (255, 255, 255).
        font_size (int, optional): font size of the text. Defaults to 25.
        font_path (_type_, optional): path to the font. Defaults to default_font.
        background_color (tuple[int, int, int], optional): Color of the background. None draws no background. Defaults to None.
        background_margin_pixel (int, optional): Margin of the background box. Defaults to 3.
        show_text (bool, optional): If true shows the bar width as text above the bar. Defaults to True.

    Returns:
        ImageSequenceSource: Rendered image sequence
    """

    # create pint quantities (values and units)
    bar_width = ureg.Quantity(bar_width)
    bar_height = ureg.Quantity(bar_height)
    size_of_pixel = ureg.Quantity(size_of_pixel)

    # load font
    font = ImageFont.truetype(font_path, font_size)

    # compute width and height of the scale bar in pixels (we need to round here)
    bar_pixel_width = int(
        np.round((bar_width / size_of_pixel).to_base_units().magnitude)
    )
    bar_pixel_height = int(
        np.round((bar_height / size_of_pixel).to_base_units().magnitude)
    )

    # extract position
    xstart, ystart = xy_position

    images = []

    for image in tqdm(image_source, desc="Render scale bar..."):

        # do we have a wrapped image?
        is_wrapped = isinstance(image, BaseImage)

        # unwrap if necessary
        if is_wrapped:
            image = image.raw

        image = np.copy(image)

        # compute text size
        text = f"{bar_width:~P}"
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # get size of text
        left, top, right, bottom = draw.textbbox((xstart, ystart), text, font=font)

        text_width = right - left
        text_height = bottom - top

        if background_color:
            cv2.rectangle(
                image,
                (xstart - background_margin_pixel, ystart + background_margin_pixel),
                (
                    xstart + bar_pixel_width + background_margin_pixel,
                    ystart
                    - text_height
                    - bar_pixel_height
                    - 5
                    - background_margin_pixel,
                ),
                background_color,
                -1,
            )

        # draw scale bar
        cv2.rectangle(
            image,
            (xstart, ystart),
            (xstart + bar_pixel_width, ystart - bar_pixel_height),
            (255, 255, 255),
            -1,
        )

        if show_text:
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)

            # draw text centered and with distance to the scale bar
            draw.text(
                (
                    xstart + bar_pixel_width / 2 - text_width / 2,
                    ystart - text_height - bar_pixel_height - 10,
                ),
                text,
                fill=color,
                font=font,
            )

            # convert PIL image back to numpy
            image = np.array(img_pil)

        images.append(image)

    # combine all images
    image_stack = np.stack(images)

    if isinstance(ImageSequenceSource, np.ndarray):
        # return as raw numpy stack
        return image_stack
    else:
        # return as sequence source again
        return InMemorySequenceSource(image_stack)


def render_time(
    image_source: ImageSequenceSource,
    xy_position: tuple[int],
    timepoints: list[pint.Quantity | timedelta],
    time_format="{H:02}h {M:02}m",
    color=(255, 255, 255),
    font_size=25,
    font_path=default_font,
    background_color: tuple[int, int, int] = None,
    background_margin_pixel=3,
) -> ImageSequenceSource:
    """Draw time onto images

    Args:
        image_source (ImageSequenceSource): image sequence of the time-lapse
        xy_position (tuple[int]): lower left xy position of the formatted time text
        timepoints (list[pint.Quantity  |  timedelta]): timepoints of the individual frames
        time_format (str, optional): Timeformat for rendering the time to the images. Defaults to "{H:02}h {M:02}m".
        color (tuple, optional): Color of the time text. Defaults to (255, 255, 255).
        font_size (int, optional): Fontsize of the time text. Defaults to 25.
        font_path (_type_, optional): Path to the rendering font. Defaults to default_font.
        background_color (tuple[int, int, int], optional): Color of the background box. None does not draw any background box. Defaults to None.
        background_margin_pixel (int, optional): Margin of the background box. Defaults to 3.

    Returns:
        ImageSequenceSource: Rendered image sequence
    """

    # load font
    font = ImageFont.truetype(font_path, font_size)

    images = []

    for image, timepoint in zip(tqdm(image_source, desc="Render time..."), timepoints):

        if isinstance(timepoint, pint.Quantity):
            timepoint = timedelta(seconds=float(timepoint.to(ureg.seconds).magnitude))

        # do we have a wrapped image?
        is_wrapped = isinstance(image, BaseImage)

        # unwrap if necessary
        if is_wrapped:
            image = image.raw

        image = np.copy(image)

        # convert to pillow image
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        time_text = strfdelta(timepoint, fmt=time_format)

        if background_color:
            # get size of text
            left, top, right, bottom = draw.textbbox(xy_position, time_text, font=font)

            text_width = right - left
            text_height = bottom - top

            x, y = xy_position

            cv2.rectangle(
                image,
                (x - background_margin_pixel, y - background_margin_pixel),
                (
                    x + text_width + background_margin_pixel,
                    y + text_height + background_margin_pixel + 5,
                ),
                background_color,
                -1,
            )

            # convert to pillow image
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

        # draw on image
        draw.text(xy_position, time_text, fill=color, font=font)

        # convert PIL image back to numpy
        image = np.array(pil_image)

        images.append(image)

    # combine all images
    image_stack = np.stack(images)

    if isinstance(ImageSequenceSource, np.ndarray):
        # return as raw numpy stack
        return image_stack
    else:
        # return as sequence source again
        return InMemorySequenceSource(image_stack)
