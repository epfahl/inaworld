"""Main driving script.

Notes
-----
* Genres for a given movie are vectorized as a binary array with a length equal
  to the total number of genres, where 1 indicates the presence of the genre.
* Summaries are tokenized such that tokens have no non-alphabetic characters
  (numbers, punctuation, etc.).
* Summaries are vectorized using the unigram bag-of-words model and the
  TF-IDF (term frequency, inverse document frequency) weighting for each token.
"""

from sklearn.svm import LinearSVC

from . import data as data_loader
from . import utils
from . import vectors
from . import filters
from . import learn

DEFAULT_DATA_PATH = 'movie_data.csv'
MIN_GENRE_COUNT = 2
TEST_SIZE = 0.25
BINARY_CLASSIFIER = LinearSVC
STRATIFY_SPLIT = False


def load_and_filter(path=None, min_genre_count=MIN_GENRE_COUNT):
    """Load data from file, filter for rows with non-empty genres and
    summaries, and return genre tokens, genre vectors, and an movie summaries
    after filtering for genres that appear at least a minimum number of times
    across the corpus.

    Returns
    -------
    {
        'genre_tokens': <filtered array of genre tokens>,
        'genre_vectors': <filtered matrix of genre indicator vectors>,
        'summaries': <filtered array of movie summaries>
    }
    """

    if path is None:
        path = utils.local_filepath(DEFAULT_DATA_PATH)

    sg = data_loader.load_summaries_genres(path)
    gv = vectors.genres(sg['genres'])
    gf = filters.genres_and_movies(gv['vectors'], min_genre_count)

    return {
        'genres': sg['genres'][gf['movies']],
        'genre_tokens': gv['tokens'][gf['genres']],
        'genre_vectors': gv['vectors'][:, gf['genres']][gf['movies'], :],
        'summaries': sg['summaries'][gf['movies']]}


def train(
    summaries, genre_vectors,
    test_size=TEST_SIZE,
    binary_classifier=BINARY_CLASSIFIER,
    stratify_split=STRATIFY_SPLIT,
    **binary_classifier_parms
):
    """Given arrays of movie summaries and genre vectors, split data into
    training and validation sets, and instantiate and train a classifier.

    Returns
    -------
    {
        'split_data': <dict of training/test data for inputs and outputs>,
        'clf': <trained classifier object>
    }

    Notes
    -----
    * Both the pipeline and data splitting functions have parameters with
      default values that are not exposed here.  See the learn module for
      details.
    """
    data = learn.split_data(
        summaries, genre_vectors,
        test_size=test_size, stratify_split=STRATIFY_SPLIT)
    clf = learn.pipeline(binary_classifier(**binary_classifier_parms))
    clf.fit(data['x_train'], data['y_train'])
    return {'split_data': data, 'clf': clf}
