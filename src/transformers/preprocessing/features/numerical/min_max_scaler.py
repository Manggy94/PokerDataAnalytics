import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X: pd.DataFrame, y=None):
        self.min = X.min()
        self.max = X.max()
        return self

    def transform(self, X: pd.DataFrame):
        X = (X - self.min) / (self.max - self.min)
        X = X.fillna(0)
        return X