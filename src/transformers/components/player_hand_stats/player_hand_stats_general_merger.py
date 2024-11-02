import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandStatsGeneralMerger(BaseEstimator, TransformerMixin):
    def __init__(self, general_player_hand_stats: pd.DataFrame):
        self.general_player_hand_stats = general_player_hand_stats\
            .drop(columns=["player", "hand_history"])\

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.general_player_hand_stats, how="left", left_on="general_stats", right_on="id", suffixes=("", "_general"))\
            .drop(columns=["general_stats", "id_general"])
