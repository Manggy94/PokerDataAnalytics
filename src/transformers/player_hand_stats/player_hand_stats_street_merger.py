import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandStatsStreetMerger(BaseEstimator, TransformerMixin):
    def __init__(self, street_player_hand_stats: pd.DataFrame, street_name: str):
        self.street = street_name
        self.street_player_hand_stats = street_player_hand_stats \
            .drop(columns=["player", "hand_history"]) \
            .rename(columns={col: f"{self.street}_{col}" for col in street_player_hand_stats.columns if col != "id"})

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.street_player_hand_stats, how="left", left_on=f"{self.street}_stats", right_on="id", suffixes=("", f"_{self.street}"))\
            .drop(columns=[f"{self.street}_stats", f"id_{self.street}"])
