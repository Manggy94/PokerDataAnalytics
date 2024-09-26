import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PositionsMerger(BaseEstimator, TransformerMixin):

    def __init__(self, positions: pd.DataFrame):
        self.positions = positions

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.positions, how="left", left_on="position", right_on="position_id", suffixes=("", "_position"))\
            .drop(columns=["position_id", "position"])\
            .rename(columns={c: f"player_{c}" for c in self.positions.columns if c != "id"})