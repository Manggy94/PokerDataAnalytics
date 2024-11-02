import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CombosCardsMerger(BaseEstimator, TransformerMixin):
    """
    This transformer merges the cards information to the combos.
    """
    def __init__(self, cards: pd.DataFrame):
        self.cards = cards.drop(columns=["name", "symbol"])

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        first_cards = self.cards.rename(columns={c: f"first_card_{c}" for c in self.cards.columns if c != "id"})
        second_cards = self.cards.rename(columns={c: f"second_card_{c}" for c in self.cards.columns if c != "id"})
        return X\
            .drop(columns=["symbol"])\
            .merge(first_cards, how='left', left_on='first_card', right_on='id', suffixes=('', '_card1'))\
            .drop(columns=['id_card1', 'first_card'])\
            .merge(second_cards, how='left', left_on='second_card', right_on='id', suffixes=('', '_card2'))\
            .drop(columns=['id_card2', 'second_card'])\
            .rename(columns={"first_card_short_name": "first_card",
                             "second_card_short_name": "second_card"})