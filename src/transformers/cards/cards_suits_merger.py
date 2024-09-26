import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CardsSuitsMerger(BaseEstimator, TransformerMixin):

        def __init__(self, suits: pd.DataFrame):
            self.suits = suits

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            suits = self.suits.drop(columns=["name", "symbol"])
            return X\
                .merge(suits, how="left", left_on="suit", right_on="id", suffixes=("", "_suit")) \
                .drop(columns=["id_suit", "suit"])\
                .rename(columns={"short_name_suit": "suit"})