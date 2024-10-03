import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandHistoriesFlopsMerger(BaseEstimator, TransformerMixin):

    def __init__(self, flops: pd.DataFrame):
        self.flops = flops.rename(columns={c: f"flop_{c}" for c in flops.columns if c != "id"})

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.flops, how="left", left_on="flop", right_on="id", suffixes=("", "_flop"))\
            .drop(columns=["id_flop", "flop"])