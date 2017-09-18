"""Functions to split into training and validation sets, and train a model.
"""

import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

SUMMARY_TOKEN_PATTERN = r"\b[a-zA-Z]+\b"
MAX_DOC_FREQ = 0.25


def pipeline(
    binary_classifier,
    max_doc_freq=MAX_DOC_FREQ,
    summary_token_pattern=SUMMARY_TOKEN_PATTERN,
):
    """Given an sklearn binary classifier and other application-specific
    parameters, return an sklearn transformation and classification Pipeline
    object with fit and predict methods.

    Notes
    -----
    * This pipeline is specifically for multi-label classification using the
      One-vs-All model.
    """
    tfidf = TfidfVectorizer(
        max_df=max_doc_freq,
        token_pattern=summary_token_pattern,
        stop_words='english')
    clf = OneVsRestClassifier(binary_classifier, n_jobs=-1)
    return Pipeline([('tfidf', tfidf), ('clf', clf)])


def split_data(x, y, test_size=0.25, stratify_split=True):
    """Given an array of movie summaries and a sparse matrix of genre indicator
    vectors, return a random training/test split of the data.

    Parameters
    ----------
    x: array of inputs
    y: array (or sparse matrix) of outputs
    test_size: fraction of data used for validation
    stratify: if True, apply stratification to ensure train and test sets have
        similar class distributions

    Returns
    -------
    {
        'x_train': <input training data>,
        'x_test': <input validation data>,
        'y_train': <output training data>,
        'y_test': <ouput validation data>,

    }

    Notes
    -----
    * Stratification doesn't seem to work when one of the inputs is a sparse
      matrix (seems like a bug).  The genre_vectors matrix must be converted
      to a dense array before stratification is applied.
    """

    xx = x.copy()
    yy = y.copy()
    stratify = None
    if stratify_split:
        if isinstance(xx, scipy.sparse.csr.csr_matrix):
            xx = xx.toarray()
        if isinstance(yy, scipy.sparse.csr.csr_matrix):
            yy = yy.toarray()
        stratify = yy

    xtrain, xtest, ytrain, ytest = train_test_split(
        xx, yy, test_size=test_size, stratify=stratify)

    return {
        'x_train': xtrain,
        'x_test': xtest,
        'y_train': ytrain,
        'y_test': ytest}
