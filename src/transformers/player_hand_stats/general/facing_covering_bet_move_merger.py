import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FacingCoveringBetMoveMerger(BaseEstimator, TransformerMixin):
    def __init__(self, action_moves: pd.DataFrame):
        self.action_moves = action_moves

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.action_moves, how="left", left_on="facing_covering_bet_move", right_on="id", suffixes=("", "_move"))\
            .drop(columns=["facing_covering_bet_move", "id_move", "name", "verb", "is_vpip_move", "is_call_move", "is_bet_move"])\
            .rename(columns={"symbol": "facing_covering_bet_move"})