import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandStatsActionStreetMerger(BaseEstimator, TransformerMixin):

        def __init__(self, streets: pd.DataFrame):
            self.streets = streets

        def fit(self, X, y=None):
            self.street_columns = [col for col in X.columns if "street" in col]
            return self

        def transform(self, X: pd.DataFrame):
            for col in self.street_columns:
                X = X\
                    .merge(self.streets, how="left", left_on=col, right_on="id", suffixes=("", f"_{col}"))\
                    .drop(columns=[col, f"id_{col}", "name", "parsing_name", "short_name", "is_preflop"])\
                    .rename(columns={"symbol": col})
            return X