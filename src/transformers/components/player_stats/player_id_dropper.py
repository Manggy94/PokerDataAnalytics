import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PlayerIdDropper(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .drop(columns='id')\
