

import logging
from typing import List
import tqdm

from acia.base import ImageSequenceSource, Overlay, Processor, Contour


class OnlineModel(Processor):
    '''
        The model is not running locally on the computer but in a remote location
    '''
    def __init__(self, url: str, port: int = None, username=None, password=None):
        '''
            url: remote model executer
            username: username
            password: password
        '''
        self.url = url
        self.port = port
        self.username = username
        self.password = password

        # try to autodetermine port
        if not self.port:
            if self.url.startswith('http'):
                self.port = 80
            elif self.url.startswith('https'):
                self.port = 443
            else:
                logging.warn('Could not determine port! Did you specify "http://" or "https://" at the beginning of your url?')
            

    def predict(self, source: ImageSequenceSource, params = {}):
        import requests
        from io import BytesIO
        from PIL import Image

        contours = []

        # iterate over images from image source
        for frame,image in tqdm.tqdm(enumerate(source)):

            # convert image into a binary png stream
            byte_io = BytesIO()
            Image.fromarray(image).save(byte_io, 'png')
            byte_io.seek(0)

            # pack this into form data
            multipart_form_data = {
                'data': ('data.png', byte_io, 'image/png'),
            }

            # send a request to the server
            response = requests.post(self.url, files=multipart_form_data, params=params)

            # raise an error if the response is not as expected
            if response.status_code != 200:
                raise ValueError('HTTP request for prediction not successful. Status code: %d' % response.status_code)

            body = response.json()

            content = body

            for detection in content:
                label = detection['label']
                contour_lists = detection['contours'][0]
                contour = list(zip(contour_lists['x'], contour_lists['y']))
                score = detection['score']

                contours.append(Contour(contour, score, frame))

        return Overlay(contours)

    @staticmethod
    def parseContours(response_body) -> List[Contour]:
        pass
