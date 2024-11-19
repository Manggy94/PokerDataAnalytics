import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: pd.DataFrame, y=None):
        self.mean = X.mean()
        self.std = X.std()
        return self

    def transform(self, X: pd.DataFrame):
        X = (X - self.mean) / self.std
        return X