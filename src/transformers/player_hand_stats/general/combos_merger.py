import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CombosMerger(BaseEstimator, TransformerMixin):

        def __init__(self, combos: pd.DataFrame):
            self.combos = combos

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X = X\
                .merge(self.combos, how="left", left_on="combo", right_on="id", suffixes=("", "_combo"))\
                .drop(columns=["id_combo", "combo"])\
                .rename(columns={c: f"player_combo_{c}" for c in self.combos.columns if c != "id"})\
                .rename(columns={"player_combo_short_name": "player_combo"})
            return X