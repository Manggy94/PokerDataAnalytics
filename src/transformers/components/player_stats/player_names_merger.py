import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PlayerNamesMerger(BaseEstimator, TransformerMixin):
    def __init__(self, player_names: pd.DataFrame, player_id_col: str):
        self.player_names = player_names
        self.player_id_col = player_id_col

    def fit(self, X: pd.DataFrame, y=None):
        return self


    def transform(self, X: pd.DataFrame):
        X = X.\
            merge(self.player_names, how='left', left_on=self.player_id_col, right_on='id', suffixes=("", "_player_name"))\
            .drop(columns=['id_player_name'])
        return X