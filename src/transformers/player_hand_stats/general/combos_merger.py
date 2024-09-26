import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CombosMerger(BaseEstimator, TransformerMixin):

        def __init__(self, combos: pd.DataFrame):
            self.combos = combos

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            return X\
                .merge(self.combos, how="left", left_on="combo", right_on="combo_id", suffixes=("", "_combo"))\
                .drop(columns=["combo_id", "combo"])\
                .rename(columns={c: f"player_{c}" for c in self.combos.columns})