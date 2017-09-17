"""Tokenize a document.

*** NLTK tokenization and this module have been deprecated in favor of a
sklearn-based solution.  However, NLTK may offer more options for tokenization,
stemming, etc., this module is retained for future reference.
"""

import re
import nltk
import toolz as tz

re_not_alpha = re.compile('[^a-zA-Z]')
STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def is_alpha(tt):
    """Given a POS tagged token (<token>, <pos>), return True if the token has
    only alphabetic characters (i.e., no punctuation or numbers).
    """
    return not bool(re_not_alpha.search(tt[0]))


def not_proper(tt):
    """Given a POS tagged token (<token>, <pos>), return True if the token is
    not tagged as a proper noun ('NNP').
    """
    return (tt[1] != 'NNP')


def not_stopword(tt):
    """Given a POS tagged token (<token>, <pos>), return True if the token is
    not a stopword.
    """
    return (tt[0] not in STOPWORDS)


def lower(tt):
    """Given a POS tagged token (<token>, <pos>), return
    (<token>.lower(), <pos>).
    """
    return (tt[0].lower(), tt[1])


def stem(tt):
    """Given a POS tagged token (<token>, <pos>), return
    (<stemmed token>, <pos>).
    """
    return (nltk.stem.lancaster.LancasterStemmer().stem(tt[0]), tt[1])


def remove_pos(tt):
    """Given a POS tagged token (<token>, <pos>), return only the token.
    """
    return tt[0]


def tokenize(doc, with_stem=False):
    """Given a document string, return a list of tokens.
    """
    pipeline = [
        (filter, is_alpha),
        (filter, not_proper),
        (map, lower),
        (filter, not_stopword)]
    if with_stem:
        pipeline += [(map, stem)]
    pipeline += [(map, remove_pos)]
    return list(tz.thread_last(
        nltk.tag.pos_tag(nltk.tokenize.word_tokenize(doc)),
        *pipeline))
