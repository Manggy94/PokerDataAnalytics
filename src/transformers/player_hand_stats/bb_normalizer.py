import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BBNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X):
        self.value_columns = [col for col in X.columns if "stack" in col]

    def transform(self, X):
        for col in self.value_columns:
            X[col] = X[col] / X["bb"]
        return X