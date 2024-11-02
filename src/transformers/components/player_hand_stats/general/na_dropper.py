import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NaDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.dropna(subset=["position"])