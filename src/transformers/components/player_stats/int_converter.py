import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IntConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.int_columns = None
        self.int_keywords = ["cnt"]

    def fit(self, X: pd.DataFrame, y=None):
        self.int_columns = [col for col in X.columns if any(keyword in col for keyword in self.int_keywords)]
        return self

    def transform(self, X: pd.DataFrame):
        X[self.int_columns] = X[self.int_columns].fillna(0).astype("uint32")
        return X