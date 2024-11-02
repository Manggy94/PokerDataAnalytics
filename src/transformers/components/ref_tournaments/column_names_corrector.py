import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnNamesCorrector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.column_names = {
            "name_speed": "speed",
            "name_type": "type"
        }

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        ref_tournaments = X.rename(columns=self.column_names)
        return ref_tournaments.rename(columns={c: f"ref_tournament_{c}" for c in ref_tournaments.columns})