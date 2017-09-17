"""Functions for filtering genre and summary data based on rudimentary
statistics.
"""

import numpy as np


def genre_counts(gv):
    """Given a matrix of movie genre indicator vectors, return an array of
    genre token counts across the corpus.
    """
    return np.array(gv.sum(axis=0))[0]


def _genre_filter(gc, min_count):
    """Given an array of genre token counts, return a boolean array that
    indicates genres with counts greater than or equal to a given minimum
    count.
    """
    return (gc >= min_count)


def _movie_filter(gv, gf):
    """Given the matrix of genre vectors and a boolean genre filter array,
    return a boolean filter array that indicates which movies have at least one
    genre after application of the genre filter.
    """
    return (np.array(gv[:, gf].sum(axis=1)) > 0).reshape(-1,)


def genres_and_movies(gv, min_genre_count):
    """Given a matrix of genre vectors, return boolean filter arrays that
    indicate which genres have at least a threshold count across the corpus,
    and which movies have at least one genre with at least the threshold count.

    Returns
    -------
    {
        'genres': <boolean array to filter genres>,
        'movies': <boolean array to filter movies>
    }
    """
    gf = _genre_filter(genre_counts(gv), min_genre_count)
    mf = _movie_filter(gv, gf)
    return {'genres': gf, 'movies': mf}
