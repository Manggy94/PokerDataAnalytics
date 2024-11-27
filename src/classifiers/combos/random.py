import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class RandomCombosClassifier(BaseEstimator, ClassifierMixin):
    """"""
    def __init__(self):
        self.classes_ = None
        self.probabilities_ = None
        self.n_samples = None
        self.n_classes = None

    def fit(self, X, y):
        self.classes_ = y.columns
        self.probabilities_ = y.mean(axis=0)
        return self

    def predict(self, X):
        values = np.random.choice(a=self.probabilities_.index, size=X.shape[0], p=self.probabilities_.values)
        empty_predictions = pd.DataFrame(np.zeros((X.shape[0], len(self.classes_))), columns=self.classes_)
        predictions_val = pd.get_dummies(values, columns=self.classes_)
        predictions = empty_predictions.add(predictions_val, fill_value=0).astype(int)
        return predictions

    def predict_proba(self, X):
        n_samples = X.shape[0]
        probabilities = np.tile(self.probabilities_, (n_samples, 1))
        predictions = pd.DataFrame(probabilities, columns=self.classes_)
        return predictions
