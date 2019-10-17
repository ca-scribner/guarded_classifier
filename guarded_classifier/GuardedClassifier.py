import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from general_utils.numpy import safe_insert_string


DEFAULT_ESTIMATOR = SVC(gamma='scale')
DEFAULT_REJECTED_CLASS_INTEGER = -1
DEFAULT_REJECTED_CLASS_STRING = "rejected"


class GuardedClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=None, min_records_in_class=30, rejected_class=None):
        """
        An sklearn-compliant meta classifier that won't return predictions for classes with few training samples

        The intent behind the classifier is to make use of data that has poorly represented classes without actually
        outwardly classifying to those classes.  The hope is that it might better represent the uncertain categories
        than an approach that just puts all those classes into a single rejected class directly (one that would
        inherently be disjointed).

        For any class trained that has more than min_records_in_class, the classifier behaves normally as expected.  For
        classes trained on fewer than min_records_in_class examples, the classifier will still try to predict to those
        classes internally but will never return those low-example predictions.  Instead, those predictions will be
        returned in the rejected_class

        Args:
            base_estimator: An sklearn-compliant multi-class estimator
            min_records_in_class (int): Number of training records below which a class will be redirected to the
                                        rejected_class
            rejected_class (str or int): Classname to assign all rejected classes to.  If None, will try to infer a
                                         suitable class name of appropriate type that matches other class types
                                         (integer or string, depending on y passed to train).  Inference, however, is
                                         not robust and specifying this is recommended.
        """
        self.min_records_in_class = min_records_in_class
        self.base_estimator = base_estimator
        self.rejected_class = rejected_class

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
            if y_pred.dtype.char == 'U':
                # If unicode class names, use safe_insert_string(), which will increase the y_pred.dtype unicode length
                # if required to avoid truncating self.rejected_class_
                y_pred = safe_insert_string(y_pred, y_pred_unguarded == name, self.rejected_class_)
            else:
                y_pred[y_pred_unguarded == name] = self.rejected_class_

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
            self.base_estimator_ = DEFAULT_ESTIMATOR

    def _validate_rejected_class(self, y):
        """
        Sets the rejected_class_ attribute, inferring a default if necessary

        rejected_class should be of a type that matches y in order to be typical of the output

        Args:
            y (array-like of shape = [n_samples]): The target values (class labels).

        Returns:
            None
        """
        # Get y's type as it will cast into a numpy array
        y_array = np.asarray(y)
        y_type = type(y_array.reshape(-1)[0])

        if self.rejected_class is None:
            # Get the type of the y array, then make a rejected_class_ that is of that type
            # This feels fragile - is there a better way?  Can asarray cause a type change that will break everything?
            try:
                temp_rejected_class = y_type(DEFAULT_REJECTED_CLASS_STRING)
            except ValueError:
                temp_rejected_class = y_type(DEFAULT_REJECTED_CLASS_INTEGER)

            if temp_rejected_class in y_array:
                raise ValueError("Default rejected_class name is present in y.  To force rejected_class to a known "
                                 "class, rejected_class must be defined explicitly (not left as default to be inferred)"
                                 " in order to avoid mistaken reassignment to existing classes")
        else:
            temp_rejected_class = self.rejected_class

        # Test if rejected_class can be injected into an array of y's without breaking
        y_like = np.empty([1], dtype=y_array.dtype)
        try:
            if y_like.dtype.char == 'U':
                y_like = safe_insert_string(y_like, 0, temp_rejected_class)
            else:
                y_like[0] = temp_rejected_class
        except ValueError:
            raise ValueError(f"rejected_class ({self.rejected_class} of type {type(self.rejected_class)} must be of"
                             f" the same type as y (of type {y_type}) or one that can automatically cast to that type")

        # Use the recast value from y_like for future rejected_class_ usage (this way it is always cast properly)
        self.rejected_class_ = y_like[0]

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
