# In a World...

... where we can automatically produce genres from movie summaries.

To get some experience with NLP and text classification, I decided to take a crack at predicting genre tags from a text summary of a film.  The focus here is on creating a working end-to-end system, not a highly polished API or a finely-tuned classifier.  That being said, the API is relatively easy to use and the default classifier performs fairly well.

It so happens that I came across a data set--scraped from Wikipedia--that lists metadata for roughly 42000 movies, including genres and summaries.  This data, included in the repo, is the default choice for classifier training and validation.  This repo has an accompanying Jupyter notebook that shows the structure of that data, explores genre tag statistics, and demonstrates the usage of this package.

The classification approach used here was heavily inspired by a [blog article](http://www.davidsbatista.net/blog/2017/04/01/document_classification/) by David Batista.  He solves essentially the same problem (genres from summaries), but with a different data set.  David's best-performing model used TF-IDF vectorization, a linear Support Vector Machine as the binary classifier, and a One-vs-Rest multi-label classifier.  I'm not ashamed of borrowing from my betters.  After all, we're all part of the same community, right?  :smiley:

## Dependencies and Installation

`Inaworld` was built and tested using the following environment and packages
* Anaconda Python distribution (Python 3.6)
* scikit-learn 0.18.1
* numpy 1.11.3
* scipy 0.18.1
* pandas 0.19.2

To install `inaworld`, you can either clone the repo,
```
git clone https://github.com/epfahl/inaworld
```
and use it from within the cloned directory, or pip install directly from github,
```
pip install git+https://github.com/epfahl/inaworld
```
The latter should handle the dependencies via `setup.py`, but be aware that this has not yet been tested in a virtualenv.

## Basic Usage

`Inaworld` has both functional and OO APIs.  It's easiest to get started with the OO API.  The followings lines, executed in a shell or Jupyter notebook, will load the canned movie data and train a classifier:

```python
>>> from inaworld import MoviesGenres
>>> mg = MovieGenres().load().train()
```

Loading and training should take a minute or less, depending your computer's horsepower.  The object `mg` gives access to various model parameters, as well as the input data, the specific training and test data used for classifier training and validation, and a `predict` method that consumes a single movie summary and returns a list of genres.

Let's pick a summary at random from the data set, predict the genres, and compare the results to the actual recorded genres:

```python
>>> idx = 12345   # there are over 40000 movies in the default data set
>>> summary = mg.data['summaries'][idx]
>>> summary
'Sam is a self-destructive, vaguely artistic New York bohemian who has recently lost his father and his long-time girlfriend. At a Halloween party he meets a mysterious, beautiful, androgynous woman named Anna. He embarks on a kinky, sex-charged relationship with her; but soon he suffers the symptoms of blood loss, and eventually he realizes that Anna is a vampire.'
>>> mg.predict(summary)
['drama', 'horror', 'psychological thriller', 'romance film']
>>>mg.data['genres'][idx]
'["Romance Film", "Drama", "Horror", "Psychological thriller"]'
```

Cool.  (Yeah, this perfect match is suspicious, but I promise I didn't cherry pick!)

## Problem Statement and Approach

The prediction of multiple genre tags (or labels) from a document is a well-trodden problem, and is an example of *multi-label* classification.  Depending on assumptions and desired sophistication, this classification problem can be attacked with a wide range of techniques.  But before addressing the classification methodology, we should say a few words about how the data is prepared.

### Vectorization

A classifier that works with text doesn't actually work with text.  Machine learning algorithms operate on numerical representations of data.  A document or a list of tags is converted to an array of numbers, a transformation known as *vectorization*.  This conversion can be accomplished in a variety of ways.  `Inaworld` vectorizes movie summaries by replacing each valid word, or, more generally, *token*, with its [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)  (term frequency-inverse document frequency) statistic.  In essence, TF-IDF records how often a token occurs in a document, inversely weighted by the log of the fraction of documents that contain the token.  TF-IDF gives less weight to tokens that are very common across the corpus.  `Inaworld` vectorizes movie summaries with the scikit-learn class `TfidfVectorizer`.

Genre tag vectorization is straightforward.  List all the unique tags in the movies data set in alphabetical order.  When a genre is present for a given movie, put a 1 at the position of the corresponding tag; otherwise, put a 0.  For each movie, there is a corresponding binary vector that represents the associated list of tags.  Scikit-learn does the heavy lifting of genre vectorization with the class [`CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).  

Genres are comma-separated words or phrases.  There is no ambiguity in parsing and listing the individual genre tags.  But tokenization isn't so simple for the movie summaries.  Should we split a summary into individual words?  Pairs or triplets of words?  Should individual words be reduced to some root grammatical form (a step known as *stemming*)?  What about super common, uninformative words like 'the' or 'you' (so-called *stop words*)?  The answers to these questions depend on a number of factors, and different choices should be evaluated when designing a high-performance classifier.  `Inaworld` applies the following specific constraints on summary tokenization:
* Only unigrams (single words) are considered.
* Words with punctuation or numbers are excluded.
* Stop words are excluded.
* By default, words that appear in more than 25% of the documents are excluded, but this parameter can be changed.

### Data Preparation

In what machine learning problem do we not spend 80% of our time on preparing the data so that it can be consumed by transformation and classification algorithms?  The movie data set used here (`movie_data.csv` in the repo) is actually in pretty decent shape, especially if we're interested only in genres and summaries.  Each row of the table has movie, title, release date, revenue, run time, a list of genres, and a summary.  

Movie genres are listed as single strings, like
```
'["Romance Film", "Drama", "Horror", "Psychological thriller"]'
```
Parsing this as a list of genres isn't a problem.  A regular-expression substitution and comma split does the job nicely.  However, there are rows with empty genre lists (411 rows, to be exact), recorded as `'[]'`.  These rows should be excised from the data set before further processing, since they offer no value for the task at hand.

Each summary is a simple string, like
```
'The film portrays an aggressive and belligerent police officer named Nariman who investigates a murder case for which an innocent man is falsely accused.'
```
Once the summary token pattern is specified (as a regular expression), along with a couple of other parameters, the `scikit-learn` class [`TfidfVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) handles the tokenization and vectorization of the summaries; no additional pre-processing is needed.  Fortunately, there are no empty summary strings in the data set, but `inaworld` has a step to remove rows with zero-length summaries, just in case.

In addition to removing rows with no genres, we might also want to remove rows where the only genres present are relatively rare.  Rare genres are unlikely to be returned in the classification, and training may be a challenge, since the training set may have genres not present in the validation set, or vice versa.  To perform this filtering, we need to count the number of times each genre appears across the corpus.  Genre counting is easiest with the vectorized form of the genre data--a matrix where each row is a binary indicator vector.  The filtering process is as follows

1. Sum down each column of the matrix to obtain the count per genre.
2. Find the columns for which the count is larger than a given threshold.
3. Create a new matrix where only the above columns are retained.
4. Find the rows in the new matrix for which there is at least one genre.

The final filter is just what we're after, and it is applied to the vectorized genres and the array of summaries to make a totally consistent data set.      

### Classification

Multi-label classification is a vast subject.  Let's shrink it by making some specific choices.  A common technique is to train an ensemble of binary classifiers, where each classifier operates on one label, and then apply a [One-vs-Rest](https://en.wikipedia.org/wiki/Multiclass_classification) strategy to make decisions about each label.  In other words, for a particular label, a binary classifier decides if the input data has the label ('One') or not ('Rest').

Which binary classifier should we choose?  In his [blog article](http://www.davidsbatista.net/blog/2017/04/01/document_classification/), David Batista studied the same problem addressed here and found that a linear [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine) yielded the highest average F1 score, just ahead of [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).  David also tried vectorization schemes other than TF-IDF, but found that TF-IDF produced the best results.  That's good enough for me!  The relevant scikit-learn classes are
* `TfidfVectorizer`
* `LinearSVC`
* `OneVsRestClassifier`

The One-vs-Rest approach makes an important implicit assumption, namely that the genres are statistically independent.  For instance, it is quite plausible that 'science fiction' and 'action' co-occur quite often.  There are approaches that exploit these conditional dependences, but these techniques were not investigated for `inaworld`.

### Training

Some care must be taken when splitting the full data set into training and validation sets, and in assessing the performance of the classifier.  More specifically, we should address both *class imbalance* over the full data set, and *distributional balance* across the split.

The genre 'drama' appears in almost half of the movies in the data set.  'Comedy', 'romance film', 'thriller', and 'action' occur much less frequently than 'drama,' but are still quite prevalent (see the figure below).  If we picked genres randomly from this set, we might see correct labels a significant fraction of the time.  While this classifier, trivial as it is, might have appreciable accuracy, it certainly won't generalize well for the genres that appear in only tens or hundreds of movies.  A discussion of *class imbalance*, the insufficiency of accuracy as a performance measure, and tactics for dealing with imbalance can be found in  [this article](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/).  `Inaworld` does not have any specific handling of class imbalance, but does dispense with accuracy in favor of precision and recall as performance metrics.

![genre_counts](genre_counts.png)

How can we be sure that the validation data set 'looks like' the training set.  That is, what can we do to encourage the validation and training sets have the same distribution of classes?  This is the goal of *stratification*, which is implemented in `inaworld` by passing the genre vectors to the `stratify` argument of the function `train_test_split` in `scikit-learn`.  Without stratification, the classification performance is significantly worse (see below).

## Performance

While the initial goal of `inaworld` was just to get something to work that didn't look too stupid, it turns out that the performance of the classifier isn't terrible.  For the default parameter set, the approximate weighted average *precision*, *recall*, and *F1 score*, across all genres, are, respectively, 0.89, 0.64, and 0.73.  When stratification is turned off, the scores are 0.53, 0.28, and 0.34--definitely worse.  Admittedly, these kinds of summary statistics are not terribly satisfying, but they are useful in comparing different choices of classifiers and parameter values.  

## Final Thoughts

There are so many ways that the tokenization, vectorization, and classification steps can be modified; David Batista investigates a few [choices](http://www.davidsbatista.net/blog/2017/04/01/document_classification/).  In fact, every aspect of the system could be changed in a variety of ways.  Here are few additional curiosities that come to mind:
* What is the impact of stemming (dog, dogs, doggy -> dog) on classification performance?
* Does removal of proper nouns, like place and person names, lead to improvements?
* How difficult would it be to implement a nearest-neighbors approach, which would naturally account for dependent genres? (`KNeighborsClassifier` in `scikit-learn` doesn't like sparse matrices for some reason.)
* How would we generate a list of semantically independent genres?  For instance, 'science fiction western' and 'screwball comedy' have clear semantic relationships with other, more primative, genres.
* How can the trained classifier be cached and serialized to reduce load time?
