import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandHistoriesTurnMerger(BaseEstimator, TransformerMixin):

    def __init__(self, cards: pd.DataFrame):
        self.cards = cards

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.cards, how="left", left_on="turn", right_on="id", suffixes=("", "_turn")) \
            .drop(columns=["id_turn", "turn", "name", "symbol"])\
            .rename(columns={c: f"turn_card_{c}" for c in self.cards.columns if c != "id"})