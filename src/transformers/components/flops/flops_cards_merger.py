import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FlopsCardsMerger(BaseEstimator, TransformerMixin):
    def __init__(self, cards: pd.DataFrame):
        self.cards = cards.drop(columns=["name", "symbol"])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cards1 = self.cards.rename(columns={c: f"first_card_{c}" for c in self.cards.columns if c!= "id"})
        cards2 = self.cards.rename(columns={c: f"second_card_{c}" for c in self.cards.columns if c!= "id"})
        cards3 = self.cards.rename(columns={c: f"third_card_{c}" for c in self.cards.columns if c!= "id"})
        return X\
            .drop(columns=["symbol"])\
            .merge(cards1, how='left', left_on='first_card', right_on='id', suffixes=('', '_card1'))\
            .drop(columns=['id_card1', 'first_card'])\
            .merge(cards2, how='left', left_on='second_card', right_on='id', suffixes=('', '_card2'))\
            .drop(columns=['id_card2', 'second_card'])\
            .merge(cards3, how='left', left_on='third_card', right_on='id', suffixes=('', '_card3'))\
            .drop(columns=['id_card3', 'third_card'])\
            .rename(columns={"first_card_short_name": "first_card",
                             "second_card_short_name": "second_card",
                             "third_card_short_name": "third_card"})
