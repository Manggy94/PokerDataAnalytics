import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NaPlayerDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.player_cols = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X[~X["player_name"].isna()]
        return X