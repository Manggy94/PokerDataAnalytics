import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CombosHandsMerger(BaseEstimator, TransformerMixin):

        def __init__(self, hands: pd.DataFrame):
            self.hands = hands

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            return X\
                .merge(self.hands, how="left", left_on="hand", right_on="id", suffixes=("", "_hand"))\
                .drop(columns=["hand_id", "hand"])\
                .rename(columns={c: f"hand_{c}" for c in self.hands.columns if c !="id"})