import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PlayerStatsGeneralMerger(BaseEstimator, TransformerMixin):

    def __init__(self, player_general_stats: pd.DataFrame):
        self.player_general_stats = player_general_stats

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):

        return X\
            .dropna(subset=["id"])\
            .merge(self.player_general_stats, how="left", left_on="general_stats", right_on="id", suffixes=("", "_general_stats"))\
            .drop(columns=["id_general_stats", "general_stats"])\
