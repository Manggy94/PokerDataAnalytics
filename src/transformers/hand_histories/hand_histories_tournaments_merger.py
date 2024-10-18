import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandHistoriesTournamentsMerger(BaseEstimator, TransformerMixin):

    def __init__(self, tournaments: pd.DataFrame):
        self.keywords = ["id", "buy_in_total", "type", "speed", "players_range"]
        self.tournaments = tournaments\
            .drop(columns=[col for col in tournaments.columns if not any(k in col for k in self.keywords)])

    def fit(self, X, y=None):

        return self

    def transform(self, X: pd.DataFrame):
        return X.merge(self.tournaments, how="left", left_on="tournament", right_on="id", suffixes=("", "_tournament"))\
            .drop(columns=["id_tournament", "tournament"])
