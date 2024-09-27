import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandsRanksMerger(BaseEstimator, TransformerMixin):

        def __init__(self, ranks: pd.DataFrame):
            self.columns_to_drop = ["name", "short_name", "is_broadway", "is_face"]
            self.ranks = ranks.drop(columns=self.columns_to_drop)

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            return X\
                .merge(self.ranks, how="left", left_on="first_rank", right_on="id", suffixes=("", "_rank"))\
                .drop(columns=["id_rank", "first_rank"])\
                .rename(columns={"symbol": "first_rank"})\
                .merge(self.ranks, how="left", left_on="second_rank", right_on="id", suffixes=("", "_rank"))\
                .drop(columns=["id_rank", "second_rank"])\
                .rename(columns={"symbol": "second_rank"})\
