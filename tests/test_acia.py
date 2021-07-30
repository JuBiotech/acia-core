#!/usr/bin/env python

"""Tests for `acia` package."""

import pytest


from acia import acia
from acia.segm.omero.shapeUtils import make_coordinates


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

def test_coordinates():
    assert make_coordinates("20,13 34,-4") == [(20, 13), (34, -4)]
