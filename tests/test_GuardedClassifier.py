import numpy as np
import numpy.testing as nptest
from contextlib import contextmanager
import pytest

from sklearn.utils.estimator_checks import check_estimator

from guarded_classifier.GuardedClassifier import GuardedClassifier


@ pytest.mark.parametrize(
    "y, unique_expected, n_counts_expected",
    (
        ([1, 1, 1, 2, 2, 3, 3, 3, 3, 3], np.array([1, 2, 3]), {1: 3, 2: 2, 3: 5}),
        (['a', 'a', 'b', 'b', 'c', 'c', 'c', 'c'], np.array(['a', 'b', 'c']), {'a': 2, 'b': 2, 'c': 4}),
    )
)
def test_GuardedClassifier_compute_counts_1(y, unique_expected, n_counts_expected):
    gc = GuardedClassifier()
    unique, n_counts = gc._compute_counts(y)
    assert np.all(unique_expected == unique)
    assert n_counts_expected == n_counts


@pytest.mark.parametrize(
    "y, expected_inferred_type",
    (
        (np.array([['abc'], ['def']]), np.str_),
        (np.array([[1], [2]]), np.integer),
        ([['abc'], ['xyz']], np.str_),
    )
)
def test_GuardedClassifier_validate_rejected_class_1(y, expected_inferred_type):
    gc = GuardedClassifier()
    gc._validate_rejected_class(y=y)
    # Should I use isinstance or == here?  Isinstance is meant for types, but doesn't give same error feedback in pytest
    assert isinstance(gc.rejected_class_, expected_inferred_type)
    # assert expected_inferred_type == type(gc.rejected_class_)


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "X, y, min_records_in_class, n_records, expectation",
    (
        ([[0], [1]], [0, 0], 0, None, pytest.raises(ValueError)),
        ([[0], [1]], [0, 1], 1, None, does_not_raise()),
        ([[0], [1]], [0, 1], 2, None, pytest.raises(ValueError)),
        ([[0], [1], [0], [1], [0], [1]], [0, 0, 1, 2, 2, 2], 2, {0: 1, 1: 1}, pytest.raises(ValueError)),
        ([[0], [1]], [0, 1], 2, {0: 10, 1: 10}, does_not_raise()),
        ([[0], [1], [0], [1], [0], [1]], [0, 0, 1, 2, 2, 2], 2, None, does_not_raise()),
        ([[0], [1], [0], [1], [0], [1]], [0, 0, 1, 2, 2, 2], 3, None, pytest.raises(ValueError)),
    )
)
def test_GuardedClassifier_fit_too_few_classes(X, y, min_records_in_class, n_records, expectation):
    gc = GuardedClassifier(min_records_in_class=min_records_in_class)
    with expectation:
        gc.fit(X, y, n_records)


def test_GuardedClassifier_sklearn_check_estimator():
    check_estimator(GuardedClassifier)
# How should I test building of the objects themselves?  Look at sklearn for inspiration?
