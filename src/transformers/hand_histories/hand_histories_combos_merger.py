import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandHistoriesCombosMerger(BaseEstimator, TransformerMixin):

    def __init__(self, combos: pd.DataFrame):
        self.combos = combos

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.combos, how="left", left_on="hero_combo", right_on="combo_id", suffixes=("", "_hero_combo"))\
            .drop(columns=["combo_id", "hero_combo"])\
            .rename(columns={c: f"hero_{c}" for c in self.combos.columns})