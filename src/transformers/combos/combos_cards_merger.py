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
        combos = X\
            .merge(self.cards, how='left', left_on='first_card', right_on='id', suffixes=('', '_card1'))\
            .merge(self.cards, how='left', left_on='second_card', right_on='id', suffixes=('', '_card2'))
        columns_to_drop = ([c for c in combos.columns if ("id_" in c or "symbol" in c)] +
                           ["name", "name_card2"] + ["first_card", "second_card"])
        combos = combos.drop(columns=columns_to_drop)
        correction_dict = {
            "is_broadway": "is_broadway_card1",
            "is_face": "is_face_card1",
            "suit": "suit_card1",
            "rank": "rank_card1",
        }
        combos = combos.rename(columns=correction_dict)
        renaming_dict = {c: f"combo_{c}" for c in combos.columns}
        X = combos.rename(columns=renaming_dict)
        return X