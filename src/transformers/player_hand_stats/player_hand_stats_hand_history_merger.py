import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandStatsHandHistoryMerger(BaseEstimator, TransformerMixin):
        def __init__(self, hand_histories: pd.DataFrame):
            self.hand_histories = hand_histories

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            return X\
                .merge(self.hand_histories, how="left", left_on="hand_history", right_on="id", suffixes=("", f"_hand_history"))\
                .drop(columns=["hand_history", f"id_hand_history"])\
                # .rename(columns={"symbol": col})