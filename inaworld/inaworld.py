"""Main driving script.
"""

import pandas as pd

from . import utils

DEFAULT_DATA_PATH = 'movie_data.csv'


def filter_summaries_genres(df):
    """Given a DataFrame of movie genres and summaries, filter rows according
    to the minimum lengths of the genre and summary strings.
    """
    return df[
        (df['genres'].str.len() > 2) &
        (df['summary'].str.len() > 0)]


def load_summaries_genres(path):
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
