import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NaBoolFiller(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keywords = ["has_", "is_", "flag_"]
        self.bool_cools = None

    def fit(self, X: pd.DataFrame, y=None):
        self.bool_cools = [col for col in X.columns
                           if any(keyword in col for keyword in self.keywords) and "ratio" not in col]
        return self

    def transform(self, X: pd.DataFrame):
        for col in self.bool_cools:
            X[col] = X[col].fillna(False)
            X[col] = X[col].astype(bool)
        return X