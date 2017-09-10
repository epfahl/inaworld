"""Load the corpus of data.
"""

import re
import pandas as pd
import toolz as tz


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


def load(path):
    """Load the CSV data, transform into an appropriate form for exploitation,
    and return a list of dicts
    """

    def tx(d):
        return tz.merge(d, {
            'genres': strlst_to_lststr(d['genres']),
            'release_date': to_datetime(d['release_date'])})

    return list(map(tx, pd.read_csv(path).to_dict('records')))
