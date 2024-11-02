import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandHistoriesCombosMerger(BaseEstimator, TransformerMixin):

    def __init__(self, combos: pd.DataFrame):
        self.combos = combos

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X\
            .merge(self.combos, how="left", left_on="hero_combo", right_on="id", suffixes=("", "_combo"))\
            .drop(columns=["id_combo", "hero_combo"])\
            .rename(columns={c: f"hero_combo_{c}" for c in self.combos.columns if c != "id"})\
            .rename(columns={"hero_combo_short_name": "hero_combo"})
        return X