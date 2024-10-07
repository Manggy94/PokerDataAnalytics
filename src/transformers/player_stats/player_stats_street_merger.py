import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PlayerStatsStreetMerger(BaseEstimator, TransformerMixin):
    def __init__(self, street_name: str, street_stats: pd.DataFrame):
        self.street_name = street_name
        self.street_stats = street_stats\
            .drop(columns=["player"])\
            .rename(columns={col: f"{self.street_name}_{col}" for col in street_stats.columns if col != "id"})

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.street_stats, how="left", left_on=f"{self.street_name}_stats", right_on="id", suffixes=("", f"_{self.street_name}"))\
            .drop(columns=[f"{self.street_name}_stats", f"id_{self.street_name}"])