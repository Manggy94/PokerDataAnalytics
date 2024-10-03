import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandStatsMoveMerger(BaseEstimator, TransformerMixin):
    def __init__(self, action_moves: pd.DataFrame):
        self.action_moves = action_moves

    def fit(self, X, y=None):
        self.move_columns = [col for col in X.columns if "move_" in col]
        return self

    def transform(self, X: pd.DataFrame):
        print(self.action_moves)
        for col in self.move_columns:
            X = X.merge(self.action_moves, how="left", left_on=col, right_on="id", suffixes=("", f"_{col}"))\
                .drop(columns=[col, f"id_{col}", "name", "verb", "is_vpip_move", "is_call_move", "is_bet_move"])\
                .rename(columns={"symbol": col})
        return X
        # return X.merge(self.action_moves, on="ref_hand_move", how="left")