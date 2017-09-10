"""Utility functions.
"""

import os


def local_filepath(filename):
    """Return the path of a local file with name <filename>.
    """
    return os.path.join(os.path.dirname(__file__), filename)
