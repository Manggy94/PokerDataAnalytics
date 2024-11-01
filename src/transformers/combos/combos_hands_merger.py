import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CombosHandsMerger(BaseEstimator, TransformerMixin):

        def __init__(self, hands: pd.DataFrame):
            self.hands = hands\
                .rename(columns={c: f"hand_{c}" for c in hands.columns if c != "id"})

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X =  X\
                .merge(self.hands, how="left", left_on="hand", right_on="id", suffixes=("", "_hand"))\
                .drop(columns=["id_hand", "hand"])\
                .rename(columns={"hand_short_name": "hand"})
            return X

