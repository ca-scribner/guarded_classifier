# Summary

A meta multi-class classifier intended to help classification workflows that may include classes that have minimal training data. 

This classifier wraps a standard scikit-learn classifier, fitting the classifier as per usual to all classes seen in training data.  On prediction, the classifier will return predictions only for classes on which it has seen more than a threshold of training data.  Any class that has less than the threshold of training data will instead be remapped to a used defined "rejected" class.  The thought behind this is that while we do not think the classifier has enough data to confidently predict the sparse classes, it might at least know enough to pick some of the sparse class entries and prevent them from appearing as false positives in other, better known classes (which is better than just ignoring the sparse classes entirely during training). 

For example, given:

```python
threshold = 2
y_train = ['A', 'A', 'A', 'B', 'C', 'D', 'D', 'D']
```

When making the predictions, the classifier internally will predict to [A, B, C, or D] as per usual, but will return predictions only to [A, D, or 'rejected'].
