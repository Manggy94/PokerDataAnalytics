import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandStatsPreflopMerger(BaseEstimator, TransformerMixin):
    def __init__(self, preflop_player_hand_stats: pd.DataFrame):
        self.preflop_player_hand_stats = preflop_player_hand_stats \
            .drop(columns=["player"]) \
            .rename(columns={col: f"preflop_{col}" for col in preflop_player_hand_stats.columns if col != "id"})

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.preflop_player_hand_stats, how="left", left_on="preflop_stats", right_on="id", suffixes=("", "_preflop"))\
            .drop(columns=["preflop_stats", "id_preflop"])\
