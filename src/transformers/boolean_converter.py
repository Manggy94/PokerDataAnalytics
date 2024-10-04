import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BooleanConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.boolean_columns = None
        self.boolean_keywords = ["flag", "is_", "has_"]

    def fit(self, X, y=None):
        self.boolean_columns = [col for col in X.columns if any(keyword in col for keyword in self.boolean_keywords)]
        return self

    def transform(self, X: pd.DataFrame):
        X[self.boolean_columns] = X[self.boolean_columns].astype(bool)
        return X