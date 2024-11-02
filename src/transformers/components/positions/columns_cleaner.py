import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PositionsColumnsCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = ["short_name", "symbol", "preflop_order", "postflop_order"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.columns_to_drop).rename(columns={c: f"position_{c}" for c in X.columns})
