import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


class GuardedClassifier(BaseEstimator, ClassifierMixin):
    # Should this be a module level global?  Hard coded in _validate_estimator?  Or is this ok as is?
    DEFAULT_ESTIMATOR = SVC()
    DEFAULT_REJECTED_CLASS = -1

    def __init__(self, base_estimator=None, min_records_in_class=30, rejected_class=None):
        self.min_records_in_class = min_records_in_class
        self.base_estimator = base_estimator
        self.rejected_class = rejected_class

        # # Shorthand for setting all params using inspect
        # # Pulled from danielhnyk.cz
        # import inspect
        # args, _, _, values = inspect.getargvalues(inspect.currentframe())
        # values.pop("self")
        # for arg, val in values.items():
        #     setattr(self, arg, val)

    def fit(self, X, y, n_records=None):
        """
        Fit the base_estimator with X and y, setting uncertain_classes_ to the classes with n_records < threshold

        n_records can be defined externally through arguments or computed internally (where n_records will be found by
        counting occurrences in y)

        Args:
            X ({array-like, sparse matrix} of shape = [n_samples, n_features]): The training input samples.
                Sparse matrix can be CSC, CSR, COO, DOK, or LIL. COO, DOK, and LIL are converted to CSR.
            y (array-like of shape = [n_samples]): The target values (class labels).
            n_records (dict, mappable): (OPTIONAL) Mappable of {class_name:n_records_in_class}.  If specified, this
                                        defines the n_records values that are compared to threshold for each class when
                                        deciding whether a class can be returned as a prediction.
                                        Must cover all classes in y.

        Returns:
            None
        """
        self._validate_estimator()

        self._validate_rejected_class(y)

        y_unique, n_records_in_y = self._compute_counts(y)
        if n_records is None:
            self.n_records_ = n_records_in_y
        else:
            self.n_records_ = n_records

        self.rejected_classes_ = [name for name in self.n_records_.keys() if self.n_records_[name] < self.min_records_in_class]
        self.n_included_classes_ = len(y_unique) - len(self.rejected_classes_)
        if not (self.n_included_classes_ > 1):
            raise ValueError(f"The number of classes with more than min_records_in_class records has to be greater than"
                             f" 1; got {self.n_included_classes_} classes with more than {self.min_records_in_class} "
                             f"records")

        self.base_estimator_.fit(X, y)

    def predict(self, X, return_unguarded=False):
        """
        Returns predictions on X

        Predictions returned are guarded from being part of a rejected class (number of examples less than
        min_records_in_class).

        Args:
            X ({array-like, sparse matrix} of shape = [n_samples, n_features]): The training input samples.
                Sparse matrix can be CSC, CSR, COO, DOK, or LIL. COO, DOK, and LIL are converted to CSR.
            return_unguarded (bool): If True, function returns an additional np.array of the unguarded predictions

        Returns:
            (np.array): Predicted classes of shape [n_samples], with all predictions that the base_estimator assigned to
                        a guarded class (n_records < min_records_in_class) reassigned to rejected_class
            (OPTIONAL) (np.array): If return_unguarded==True, predicted classes of shape [n_samples]
        """
        y_pred_unguarded = self.predict_unguarded(X)
        y_pred = y_pred_unguarded.copy()

        for name in self.rejected_classes_:
            y_pred[y_pred_unguarded == name] = self.rejected_class

        if return_unguarded:
            return y_pred, y_pred_unguarded
        else:
            return y_pred

    def predict_unguarded(self, X):
        """
        Returns unguarded predictions on X

        Predictions returned are exactly as predicted by the base_estimator and are not guarded by the
        min_records_in_class threshold

        Args:
            X ({array-like, sparse matrix} of shape = [n_samples, n_features]): The training input samples.
                Sparse matrix can be CSC, CSR, COO, DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns:
            (np.array): Predicted classes of shape [n_samples]

        """
        return self.base_estimator_.predict(X)

    def _validate_estimator(self):
        """
        Sets the base_estimator_ attribute, using a default if necessary

        Returns:
            None
        """
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = self.DEFAULT_ESTIMATOR

    def _validate_rejected_class(self, y):
        """
        Sets the rejected_class_ attribute, inferring a default if necessary

        rejected_class should be of a type that matches y in order to be typical of the output

        Args:
            y (array-like of shape = [n_samples]): The target values (class labels).

        Returns:
            None
        """
        if self.rejected_class is None:
            # Get the type of the y array, then make a rejected_class_ that is of that type
            # This feels fragile - is there a better way?  Can asarray cause a type change that will break everything?
            y_type = type(np.asarray(y).reshape(-1)[0])
            self.rejected_class_ = y_type(self.DEFAULT_REJECTED_CLASS)
        else:
            self.rejected_class_ = self.rejected_class

    @staticmethod
    def _compute_counts(y):
        """
        Returns a dict of {unique_element_of_y: n_occurrences_of_element} for all elements of y

        Args:
            y (list-like): Iterable of one or more labels or values

        Returns:
            Tuple of:
                (np.array): Unique classes in y
                (dict): {unique_element_of_y: n_occurrences_of_element}
        """
        uniques, counts = np.unique(y, return_counts=True)
        return uniques, {record: count for record, count in zip(uniques, counts)}