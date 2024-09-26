import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CardsRanksMerger(BaseEstimator, TransformerMixin):

        def __init__(self, ranks: pd.DataFrame):
            self.ranks = ranks

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            ranks = self.ranks.drop(columns=["name", "is_broadway", "is_face", "symbol"])
            return X\
                .merge(ranks, how="left", left_on="rank", right_on="id", suffixes=("", "_rank")) \
                .drop(columns=["id_rank", "rank"])\
                .rename(columns={"short_name_rank": "rank"})