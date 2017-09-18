# In a World...

... where we can automatically produce genres from movie summaries.

As an exploration of NLP and text classification, I decided to take a crack at predicting genre tags from a text summary of a film.  The focus here is on creating a working end-to-end system, not a highly polished API or a finely-tuned classifier.  That being said, the API is relatively easy to use and the default classifier performs fairly well.

It so happens that I came across a data set--scraped from Wikipedia--that lists metadata for roughly 42000 movies, including genres and summaries.  This data, included in the repo, is the default choice for classifier training and validation.  This repo has an accompanying Jupyter notebook that shows the structure of that data, explores genre tag statistics, and demonstrates the usage of this package.

## Basic Usage

`Inaworld` has both functional and OO APIs.  It's easiest to get started with the OO API.  The followings lines, executed in a shell or Jupyter notebook, will load some canned movie data and train a classifier:

```python
>>> from inaworld import MoviesGenres
>>> mg = MovieGenres().load().train()
```

The loading and training sequence should take a minute or less.  The object `mg` gives access to various parameters, as well as the input data, the specific training and test data used for classifier training and validation, and a `predict` method that consumes a single movie summary and returns a list of genres.

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

A classifier that works with text doesn't actually work with text.  Machine learning algorithms operate on numerical representations of data.  A document or a list of tags is converted to an array of numbers--a transformation known as *vectorization*.  This conversion can be accomplished in a variety of ways.  `Inaworld` vectorizes movie summaries by replacing each valid word, or token, with its [TF-IDF](http://docs.python-cerberus.org/)  (term frequency-inverse document frequency) statistic.  In essence, TF-IDF records how often a word occurs in a document, inversely weighted by the number of documents that feature the word.  TF-IDF gives less weight to words that are very common across the corpus.  Movie summary vectorization is achieved with the scikit-learn class `TfidfVectorizer`.

Genre tag vectorization is straightforward.  List all the unique tags in the movies data set in alphabetical order.  When a genre is present for a given movie, put a 1 at the position of the corresponding tag; otherwise, put a 0.  For each movie, there is a corresponding binary vector that represents the associated list of tags.  Scikit-learn does the heavy lifting of genre vectorization with the class `CountVectorizer`.  

Genres are comma-separated words or phrases.  There is no ambiguity in parsing and listing the individual genre tags.  But the situation isn't so simple for the movie summaries.  Should we split a summary into individual words?  Pairs or triplets of words?  Should individual words be reduced to some root grammatical form (known as *stemming*)?  What about super common, uninformative words like 'the' or 'you' (so-called *stop words*)?  The answers to these questions depend on a number of factors, and different choices should be evaluated when designing a high-performance classifier.  `Inaworld` applies the following specific constraints on summary tokenization:
* Only unigrams (single words) are considered.
* Words with punctuation or numbers are excluded.
* Stop words are excluded.
* By default, words that appear in more than 25% of the documents are excluded, but this parameter can be changed.

### Classification

* one-vs-rest plus binary classifier
* neglects conditional dependence of genres

### Data Preparation

* rare genres are removed
* movies with only rare genres are removed


## Final Thoughts

* tokenization (stemming, remove proper nouns, remove non-words [e.g., 'aaaah'])
* different binary classifiers (kernel SVM, naive bayes, logistic regression)
* entirely different approach (kNN variant to account for tag correlations)
* Not only do genres tend to show correlations, but there may also be hidden
semantic relationships.  For instance, if 'drama' is a returned tag, but the
actual tag is 'melodrama,' does that count as a miss?
* cache the trained classifier for production performance (e.g., real-time tagging).
