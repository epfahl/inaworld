"""Functions for filtering genre and summary data based on rudimentary
statistics.
"""

import numpy as np


def genre_counts(gv):
    """Given a matrix of movie genre indicator vectors, return an array of
    genre token counts across the corpus.
    """
    return np.array(gv.sum(axis=0))[0]


def genres(gc, min_count):
    """Given an array of genre token counts, return a boolean array that
    indicates genres with counts greater than or equal to a given minimum
    count.
    """
    return (gc >= min_count)


def movies(gv, gf):
    """Given the matrix of genre vectors and a boolean genre filter array,
    return a boolean filter array that indicates which movies have at least one
    genre after application of the genre filter.
    """
    return (np.array(gv[:, gf].sum(axis=1)) > 0).reshape(-1,)
