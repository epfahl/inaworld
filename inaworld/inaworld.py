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
from . import validation

DEFAULT_DATA_PATH = 'movie_data.csv'
MIN_GENRE_COUNT = 2
TEST_SIZE = 0.25
BINARY_CLASSIFIER = LinearSVC
STRATIFY_SPLIT = True


class UntrainedError(Exception):
    pass


class UnloadedError(Exception):
    pass


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


def split_and_train(
    data,
    test_size=TEST_SIZE,
    binary_classifier=BINARY_CLASSIFIER,
    stratify_split=STRATIFY_SPLIT,
    **binary_classifier_parms
):
    """Given a data payload that contains arrays of movie summaries and genre
    vectors, split data into training and validation sets, and instantiate and
    train a classifier.

    Returns
    -------
    (
        <dict of training/test data for inputs and outputs>,
        <trained classifier object with predict method>
    )

    Notes
    -----
    * Both the pipeline and data splitting functions have parameters with
      default values that are not exposed here.  See the learn module for
      details.
    """
    summaries, genre_vectors = data['summaries'], data['genre_vectors']
    data_split = learn.split_data(
        summaries, genre_vectors,
        test_size=test_size, stratify_split=stratify_split)
    clf = learn.pipeline(binary_classifier(**binary_classifier_parms))
    clf.fit(data_split['x_train'], data_split['y_train'])
    return (data_split, clf)


def predict_genres(clf, genre_tokens, summary):
    """Given a classifier object, an array of genre tokens, and a movie
    summary, return a list of genres.
    """
    return list(genre_tokens[clf.predict([summary])[0].astype(bool)])


class MovieGenres(object):

    def __init__(
        self,
        path=None,
        min_genre_count=MIN_GENRE_COUNT,
        test_size=TEST_SIZE,
        binary_classifier=BINARY_CLASSIFIER,
        stratify_split=STRATIFY_SPLIT,
        **binary_classifier_parms
    ):
        if path is None:
            self.path = utils.local_filepath(DEFAULT_DATA_PATH)
        else:
            self.path = None
        self.min_genre_count = min_genre_count
        self.test_size = test_size
        self.binary_classifier = binary_classifier
        self.stratify_split = stratify_split
        self.binary_classifier_parms = binary_classifier_parms

    def load(self):
        self.data = load_and_filter(
            path=self.path, min_genre_count=self.min_genre_count)
        return self

    def genre_counts(self):
        """Return a dict of filtered genre tokens and counts.
        """
        if getattr(self, 'data', None) is None:
            raise UnloadedError(
                "Padowan, you must first load data before you can wield its "
                "power!")
        return dict(zip(
            self.data['genre_tokens'],
            filters.genre_counts(self.data['genre_vectors'])))

    def train(self):
        """Split data into training and validation sets, and train the
        classifier.
        """
        if getattr(self, 'data', None) is None:
            raise UnloadedError(
                "Padowan, you must first load data before you can be trained "
                "in its use!")
        data_split, clf = split_and_train(
            self.data,
            test_size=self.test_size,
            binary_classifier=self.binary_classifier,
            stratify_split=self.stratify_split,
            **self.binary_classifier_parms)
        self.data_split = data_split
        self.clf = clf
        return self

    def predict(self, summary):
        """Given a movie summary, return a list of genres.  An exception is
        raised if the classifier hasn't yet been trained.
        """
        if getattr(self, 'clf', None) is None:
            raise UntrainedError(
                "Young Jedi, you must train before you can predict!")
        return predict_genres(self.clf, self.data['genre_tokens'], summary)

    def report(self):
        """Return a string report on the classifier performance using the test
        data.
        """
        return validation.report(
            self.clf,
            self.data_split['x_test'],
            self.data_split['y_test'],
            self.data['genre_tokens'])
