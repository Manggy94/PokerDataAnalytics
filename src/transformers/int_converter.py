import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IntConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.int_8_columns = None
        self.int_16_columns = None
        self.int_8_keywords = ["rank_difference", "cnt", "seat", "distance", "max", "count"]
        self.int_16_keywords = ["level_value"]

    def fit(self, X, y=None):
        self.int_8_columns = [col for col in X.columns if any(keyword in col for keyword in self.int_8_keywords)]
        self.int_16_columns = [col for col in X.columns if any(keyword in col for keyword in self.int_16_keywords)]
        return self

    def transform(self, X: pd.DataFrame):
        X[self.int_8_columns] = X[self.int_8_columns].astype("Int8")
        X[self.int_16_columns] = X[self.int_16_columns].astype("Int16")
        return X