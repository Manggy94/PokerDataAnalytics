import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SeatsCategorizer(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X['seat'] = X['seat'].astype(str)
        return X