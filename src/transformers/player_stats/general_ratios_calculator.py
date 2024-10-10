import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GeneralRatiosCalculator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ratio_columns = ["cnt_went_to_showdown", "cnt_won_hand"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X["went_to_showdown_ratio"] = X["cnt_went_to_showdown"] / X["cnt_hands_played"]
        X["won_hand_ratio"] = X["cnt_won_hand"] / X["cnt_hands_played"]
        X["confidence_ratio"] = 1-1/np.sqrt(X["cnt_hands_played"])
        return X
