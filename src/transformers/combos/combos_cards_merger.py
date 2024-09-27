import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CombosCardsMerger(BaseEstimator, TransformerMixin):
    """
    This transformer merges the cards information to the combos.
    """
    def __init__(self, cards: pd.DataFrame):
        self.cards = cards

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.cards, how='left', left_on='first_card', right_on='id', suffixes=('', '_card1'))\
            .drop(columns=['id_card1', 'first_card', 'name', 'symbol'])\
            .rename(columns={c: f"first_card_{c}" for c in self.cards.columns if c != "id"})\
            .merge(self.cards, how='left', left_on='second_card', right_on='id', suffixes=('', '_card2'))\
            .drop(columns=['id_card2', 'second_card', 'name', 'symbol'])\
            .rename(columns={c: f"second_card_{c}" for c in self.cards.columns if c != "id"})