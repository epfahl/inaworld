"""Main driving script.
"""

# import toolz as tz
import multiprocessing

from . import tokens
from . import data


def load_genres_summaries(path=None):
    return data.load(path=path)


def _tokenize(gs, with_stem=False):
    """Given a dict of genres and summary, return a dict with summary replaced
    by tokens.
    """
    return {
        'genres': gs['genres'],
        'summary': tokens.tokenize(gs['summary'], with_stem=with_stem)}


def load_genres_tokens(genres_summaries):
    """Given {'genres': List String, 'summary': String}, return
    {'genres': List String, 'summary': List String}.
    """
    return multiprocessing.Pool().map(_tokenize, genres_summaries)
