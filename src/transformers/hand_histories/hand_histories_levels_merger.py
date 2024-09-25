import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandHistoriesLevelsMerger(BaseEstimator, TransformerMixin):

    def __init__(self, levels: pd.DataFrame):
        self.levels = levels

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.levels, how="left", left_on="level", right_on="id", suffixes=("", "_level"))\
            .drop(columns=["id_level", "level"]).rename(columns={c: f"level_{c}" for c in self.levels.columns})