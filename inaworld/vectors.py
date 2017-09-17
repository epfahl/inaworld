"""Functions to vectorize movie genres and summaries.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def genres_tokenizer(gnrs):
    """Given genres as a monolithic string of comma-separated genre tokens,
    return a list of genre tokens.

    Examples
    --------
    >>> genres_tokenizer('["Action", "Space western"]')
    ['Action', 'Space western']
    """
    return re.sub('[\[\]\"]', '', gnrs).split(', ')


def genres(gnrs_ary):
    """Given an array of unprocessed genre strings, return (genre tokens array,
    boolean indicator sparse matrix).
    """
    vectorizer = CountVectorizer(binary=True, tokenizer=genres_tokenizer)
    vectors = vectorizer.fit_transform(gnrs_ary)
    return {
        'tokens': np.array(vectorizer.get_feature_names()),
        'vectors': vectors}
