import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FlopsCardsMerger(BaseEstimator, TransformerMixin):
    def __init__(self, cards: pd.DataFrame):
        self.cards = cards

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        flops = X\
            .merge(self.cards, how='left', left_on='first_card', right_on='id', suffixes=('', '_card1')) \
            .merge(self.cards, how='left', left_on='second_card', right_on='id', suffixes=('', '_card2')) \
            .merge(self.cards, how='left', left_on='third_card', right_on='id', suffixes=('', '_card3'))
        columns_to_drop = ([c for c in flops.columns if ("id_" in c or "symbol" in c)] +
                           ["name", "name_card2", "name_card3"] + ["first_card", "second_card", "third_card"])
        # Drop columns
        flops = flops.drop(columns=columns_to_drop)
        # Rename columns on card1 to match the naming convention
        correction_dict = {
            "is_broadway": "is_broadway_card1",
            "is_face": "is_face_card1",
            "suit": "suit_card1",
            "rank": "rank_card1",
        }
        flops = flops.rename(columns=correction_dict)
        # Add prefix to columns
        renaming_dict = {c: f"flop_{c}" for c in flops.columns}
        flops = flops.rename(columns=renaming_dict)
        return flops