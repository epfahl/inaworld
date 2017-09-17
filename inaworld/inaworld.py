"""Main driving script.
"""

import pandas as pd

from . import utils
from . import vectors
from . import filters

DEFAULT_DATA_PATH = 'movie_data.csv'
MIN_GENRE_COUNT = 2


def filter_summaries_genres(df):
    """Given a DataFrame of movie genres and summaries, filter rows according
    to the minimum lengths of the genre and summary strings.
    """
    return df[
        (df['genres'].str.len() > 2) &
        (df['summary'].str.len() > 0)]


def load_summaries_genres(path=None):
    """Load movie data and return (summary string array, genre string array).
    The rows are filtered to remove entries with empty genre lists and
    summaries, but the data is otherwise unprocessed.
    """

    if path is None:
        path = utils.local_filepath(DEFAULT_DATA_PATH)

    drop_cols = [
        'id', 'title', 'release_date',
        'runtime', 'box_office_revenue']
    df = filter_summaries_genres(pd.read_csv(path).drop(drop_cols, axis=1))
    return (df['summary'].values, df['genres'].values)


def vectorize_and_filter(smrs, gnrs, min_genre_count=MIN_GENRE_COUNT):
    """Given arrays of unprocessed summary and genre strings, return filtered
    arrays of tokens and vectors for summaries and genres.  Filtered is done to
    remove uncommon genres and movies that contain only uncommon genres.

    Returns
    -------

    {
        'summaries':
            {
                'tokens': <array of summary tokens>,
                'vectors': <sparse matrix of filtered summary vectors>
            },
        'genres':
            {
                'tokens': <array of filtered genre tokens>,
                'vectors': <sparse matrix of filtered genre vectors>
            }
    }

    Note
    ----
    * This feels a bit clunkly (i.e., procedural) because the genre and movie
      filters are built upon the genre vectorization and then applied to the
      genre and summary vectors.
    """
    gt, gv = vectors.genres(gnrs)
    gf = filters.genres(filters.genre_counts(gv), min_genre_count)
    mf = filters.movies(gv, gf)
    st, sv = vectors.summaries(smrs[mf])
    return {
        'summaries': {'tokens': st, 'vectors': sv},
        'genres': {'tokens': gt[gf], 'vectors': gv[:, gf][mf, :]}}
