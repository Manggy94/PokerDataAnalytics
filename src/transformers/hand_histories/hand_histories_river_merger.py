import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandHistoriesRiverMerger(BaseEstimator, TransformerMixin):

    def __init__(self, cards: pd.DataFrame):
        self.cards = cards

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X =  X\
            .merge(self.cards, how="left", left_on="river", right_on="id", suffixes=("", "_river")) \
            .drop(columns=["id_river", "river", "name", "symbol"])\
            .rename(columns={c: f"river_card_{c}" for c in self.cards.columns if c != "id"})
        return X