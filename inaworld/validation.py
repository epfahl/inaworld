"""Functions to assess classifier performance and to spot check specific
input/output pairs.
"""

from sklearn.metrics import classification_report


def report(clf, x_test, y_test, tokens):
    """Given a trained classifier with a predict method, input and output test
    data, and a tokens that are index-aligned with the columns of the output
    test data, return a string report of the classification perforamnce.
    """
    y_pred = clf.predict(x_test)
    return classification_report(y_test, y_pred, target_names=tokens)
