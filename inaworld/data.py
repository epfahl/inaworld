"""Load the corpus of data.
"""

import re
import pandas as pd
import toolz as tz

from . import utils

DEFAULT_DATA_PATH = 'movie_data.csv'


def strlst_to_lststr(genres):
    """Convert a string of list entries to a list of lowercase strings.

    Examples
    --------
    >>> strlst_to_lststr('["Banking Hijinks", "Actuarial adventure"]')
    ['banking hijinks', "actuarial adventure"]
    """
    return list(map(
        lambda g: g.strip().lower()[1:-1],
        re.sub('[\[\]]', '', genres).split(',')))


def to_datetime(date):
    """Given a string date, return a Python datetime.date.  If only the year is
    given, the date is (<year>, 1, 1).  If the result is a null type or if an
    exception is raised, None is returned.
    """
    try:
        ret = pd.to_datetime(date).to_pydatetime().date()
    except:
        ret = None
    if pd.isnull(ret):
        ret = None
    return ret


def load(path=None):
    """Load the CSV data, transform into an appropriate form for exploitation,
    and return a list of dicts

    Notes
    -----
    * Only the genres and summaries are retained.  It is straightforward to
      retain additional columns and include other transformations.
    """

    drop_cols = [
        'id', 'title', 'release_date',
        'runtime', 'box_office_revenue']

    if path is None:
        path = utils.local_filepath(DEFAULT_DATA_PATH)

    def tx(d):
        return tz.merge(d, {
            'genres': strlst_to_lststr(d['genres'])})

    return list(map(
        tx,
        (
            pd.read_csv(path)
            .drop(drop_cols, axis=1)
            .to_dict('records'))))
