import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X["id"] = X["id"].astype("uint8")
        ordered_streets = ["PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
        ordered_symbols = ["PF", "F", "T", "R", "SD"]
        X["name"] = X["name"].astype("category").cat.set_categories(ordered_streets, ordered=True)
        X["symbol"] = X["symbol"].astype("category").cat.set_categories(ordered_symbols, ordered=True)
        X["short_name"] = X["short_name"].astype("category").cat.set_categories(ordered_symbols, ordered=True)
        X["parsing_name"] = X["parsing_name"].astype("category")
        return X