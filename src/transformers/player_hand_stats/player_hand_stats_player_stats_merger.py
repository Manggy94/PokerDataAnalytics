import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PlayerHandStatsPlayerStatsMerger(BaseEstimator, TransformerMixin):

        def __init__(self, player_stats: pd.DataFrame):
            self.player_stats = player_stats\
                .rename(columns={col: f"player_{col}" for col in player_stats.columns if col != "player"})

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):

            return X.merge(self.player_stats, how="left", left_on="player", right_on="player", suffixes=("", "_stats"))
