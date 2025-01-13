"""Module for testing visualization utils module"""

import unittest
from datetime import timedelta

from acia.viz.utils import strfdelta


class TestUtils(unittest.TestCase):
    """Testing scalebar and time overlay"""

    def test_formatting(self):
        # Check that video file exists
        self.assertEqual(
            strfdelta(timedelta(days=2, minutes=22, hours=12, minutes=2, seconds=13)),
            "02d 12h 22m 13s",
        )


if __name__ == "__main__":
    unittest.main()
