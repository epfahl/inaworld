"""Functions to vectorize movie genres and summaries.

Notes
-----
* Genres for a given movie are vectorized as a binary array with a length equal
  to the total number of genres, where 1 indicates the presence of the genre.
* Summaries are tokenized such that tokens have no non-alphabetic characters
  (numbers, punctuation, etc.).
* Summaries are vectorized using the unigram bag-of-words model and the
  TF-IDF (term frequency, inverse document frequency) weighting for each token.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

SUMMARY_TOKEN_PATTERN = r"\b[a-zA-Z]+\b"
MAX_DF = 0.25


def genres_tokenizer(genres):
    """Given genres as a monolithic string of comma-separated genre tokens,
    return a list of genre tokens.

    Examples
    --------
    >>> genres_tokenizer('["Action", "Space western"]')
    ['Action', 'Space western']
    """
    return re.sub('[\[\]\"]', '', genres).split(', ')


def genres(gnrs):
    """Given an array of unprocessed genre strings, return (genre tokens array,
    boolean indicator sparse matrix).
    """
    vectorizer = CountVectorizer(binary=True, tokenizer=genres_tokenizer)
    vectors = vectorizer.fit_transform(gnrs)
    return (
        np.array(vectorizer.get_feature_names()),
        vectors)


def summaries(smrs, max_df=MAX_DF):
    """Given an array of unprocessed summary strings, return (summary tokens
    array, TF-IDF sparse matrix).
    """
    vectorizer = TfidfVectorizer(
        max_df=max_df,
        token_pattern=SUMMARY_TOKEN_PATTERN,
        stop_words='english')
    vectors = vectorizer.fit_transform(smrs)
    return (
        np.array(vectorizer.get_feature_names()),
        vectors)
