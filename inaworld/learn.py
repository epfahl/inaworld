"""Functions to preprocess vectorized data, split into training and validation
sets, and train a model.
"""

from . import filters


def preprocess(sv, gv, min_genre_count=1):
    """Given summary and genre vectorizations, return filtered matrices
    containing 1) only genres with at least the minimum count
    (well-populated genres), and 2) only movies with at least one
    well-populated genre.
    """
    gf = filters.genres(filters.genre_counts(gv), min_genre_count)
    mf = filters.movies(gv, gf)
    return (sv[mf, :], gv[:, gf])
