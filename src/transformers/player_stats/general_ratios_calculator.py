import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GeneralRatiosCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ratio_columns = ["cnt_went_to_showdown", "cnt_won_hand"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.ratio_columns:
            X[f'{col.replace("cnt", "ratio")}'] = X[col] / X['cnt_hands_played']
        return X
