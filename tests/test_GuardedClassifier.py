import numpy as np
import numpy.testing as nptest
from contextlib import contextmanager
import pytest

from sklearn.utils.estimator_checks import check_estimator

import guarded_classifier
from guarded_classifier.GuardedClassifier import GuardedClassifier


# Helpers
@contextmanager
def does_not_raise():
    yield


# Tests

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


# Test whether _validate_rejected_class detects if the inferred class overlaps classes in y
@pytest.mark.parametrize(
    "settings",
    (
        # With integer y
        {
            'y': np.array([0, 1, 2]),
            'rejected_class': None,
            'expected_rejected_class': guarded_classifier.GuardedClassifier.DEFAULT_REJECTED_CLASS_INTEGER,
            'raises': does_not_raise(),
        },
        {
            'y': np.array([0, 1, 2]),
            'rejected_class': -999,
            'expected_rejected_class': -999,
            'raises': does_not_raise(),
        },
        {
            'y': np.array([-1, 0, 1, 2]),
            'rejected_class': None,
            'expected_rejected_class': None,
            'raises': pytest.raises(ValueError),
        },
        {
            'y': np.array([-1, 0, 1, 2]),
            'rejected_class': -1,
            'expected_rejected_class': -1,
            'raises': does_not_raise(),
        },
        {
            'y': np.array([-1, 0, 1, 2]),
            'rejected_class': 'not_an_integer',
            'expected_rejected_class': None,
            'raises': pytest.raises(ValueError),
        },

        # With string y
        {
            'y': np.array(['a', 'b', 'c']),
            'rejected_class': None,
            'expected_rejected_class': guarded_classifier.GuardedClassifier.DEFAULT_REJECTED_CLASS_STRING,
            'raises': does_not_raise(),
        },
        {
            'y': np.array(['a', 'b', 'c']),
            'rejected_class': "my_rejected",
            'expected_rejected_class': "my_rejected",
            'raises': does_not_raise(),
        },
        {
            'y': np.array(['a', 'b', 'c', 'rejected']),
            'rejected_class': None,
            'expected_rejected_class': None,
            'raises': pytest.raises(ValueError),
        },
        {
            'y': np.array(['a', 'b', 'c', 'rejected']),
            'rejected_class': 'rejected',
            'expected_rejected_class': 'rejected',
            'raises': does_not_raise(),
        },
        # Works because -999 can be cast as "-999", although maybe this shouldn't be default behaviour?
        {
            'y': np.array(['a', 'b', 'c', 'rejected']),
            'rejected_class': -999,
            'expected_rejected_class': "-999",
            'raises': does_not_raise(),
        },

    )
)
def test_GuardedClassifier_validate_rejected_class_2(settings):
    gc = GuardedClassifier(rejected_class=settings['rejected_class'])

    with settings['raises']:
        gc._validate_rejected_class(y=settings['y'])
        assert settings['expected_rejected_class'] == gc.rejected_class_


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


@pytest.mark.parametrize(
    "settings",
    (
        {'X': [[0], [0], [1], [2], [2], [2]],
         'y': [0, 0, 1, 2, 2, 2],
         'y_pred': [0, 0, 1, 2, 2, 2],
         'min_records_in_class': 0,
         'rejected_class': None,
         'raises': does_not_raise()},
        {'X': [[0], [0], [1], [2], [2], [2]],
         'y': [0, 0, 1, 2, 2, 2],
         'y_pred': [0, 0, -1, 2, 2, 2],
         'min_records_in_class': 2,
         'rejected_class': None,
         'raises': does_not_raise()},
        {'X': [[0], [0], [1], [2], [2], [2]],
         'y': [0, 0, 1, 2, 2, 2],
         'y_pred': [0, 0, -999, 2, 2, 2],
         'min_records_in_class': 2,
         'rejected_class': -999,
         'raises': does_not_raise()},
        {'X': [[0], [0], [1], [2], [2], [2]],
         'y': [0, 0, 1, 2, 2, 2],
         'y_pred': [0, 0, None, 2, 2, 2],
         'min_records_in_class': 2,
         'rejected_class': "not_an_integer",
         'raises': pytest.raises(ValueError)},

        {'X': [[0], [0], [1], [2], [2], [2]],
         'y': ['a', 'a', 'b', 'c', 'c', 'c'],
         'y_pred': ['a', 'a', 'b', 'c', 'c', 'c'],
         'min_records_in_class': 0,
         'rejected_class': None,
         'raises': does_not_raise()},
        {'X': [[0], [0], [1], [2], [2], [2]],
         'y': ['a', 'a', 'b', 'c', 'c', 'c'],
         'y_pred': ['a', 'a', 'rejected', 'c', 'c', 'c'],
         'min_records_in_class': 2,
         'rejected_class': None,
         'raises': does_not_raise()},
        {'X': [[0], [0], [1], [2], [2], [2]],
         'y': ['a', 'a', 'b', 'c', 'c', 'c'],
         'y_pred': ['a', 'a', 'my_rejected', 'c', 'c', 'c'],
         'min_records_in_class': 2,
         'rejected_class': "my_rejected",
         'raises': does_not_raise()},
    )
)
def test_GuardedClassifier_predict_rejected_classes(settings):
    with settings['raises']:
        gc = GuardedClassifier(min_records_in_class=settings['min_records_in_class'],
                               rejected_class=settings['rejected_class'])
        gc.fit(settings['X'], settings['y'])
        y_pred = gc.predict(settings['X'])
        print(y_pred)
        print(settings['y_pred'])
        assert np.all(settings['y_pred'] == y_pred)


# def test_GuardedClassifier_sklearn_check_estimator():
#     check_estimator(GuardedClassifier)


# # How should I test building of the objects themselves?  Look at sklearn for inspiration?


