"""Load movie summaries and genres from file.
"""

import pandas as pd


def filter_genres_summaries(data):
    """Given a DataFrame of movie genres and summaries, filter rows according
    to the minimum lengths of the genre and summary strings.
    """
    return data[
        (data['genres'].str.len() > 2) &
        (data['summary'].str.len() > 0)]


def load_summaries_genres(path):
    """Load movie summaries and genres as separate arrays.  The rows are
    filtered to remove entries with empty genre lists and , but the genres and
    summaries are unprocessed.
    """

    drop_cols = [
        'id', 'title', 'release_date',
        'runtime', 'box_office_revenue']
    df = filter_genres_summaries(
        pd.read_csv(path)
        .drop(drop_cols, axis=1))
    return (df['summary'].values, df['genres'].values)
