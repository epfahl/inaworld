"""Functions to load data from file and apply rudimentary filtering.
"""

import pandas as pd


def filter_summaries_genres(df):
    """Given a DataFrame of movie genres and summaries, filter rows according
    to the minimum lengths of the genre and summary strings.
    """
    return df[
        (df['genres'].str.len() > 2) &
        (df['summary'].str.len() > 0)]


def load_summaries_genres(path):
    """Load movie data and return arrays of filtered, but unprocessed, movie
    genres and summaries.  The rows of the input data are filtered to remove
    entries with empty genre lists and summaries.

    Returns
    -------
    {
        'summaries': <array of summary strings>,
        'genres': <array of genre strings>
    }
    """
    drop_cols = [
        'id', 'title', 'release_date',
        'runtime', 'box_office_revenue']
    df = filter_summaries_genres(pd.read_csv(path).drop(drop_cols, axis=1))
    return {'summaries': df['summary'].values, 'genres': df['genres'].values}
