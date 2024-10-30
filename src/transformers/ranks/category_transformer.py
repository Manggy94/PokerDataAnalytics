import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        ordered_names = ["DEUCE", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "JACK", "QUEEN", "KING", "ACE"]
        ordered_symbols = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        X["id"] = X["id"].astype("int8")
        X["name"] = X["name"].astype("category").cat.set_categories(ordered_names, ordered=True)
        X["symbol"] = X["symbol"].astype("category").cat.set_categories(ordered_symbols, ordered=True)
        X["short_name"] = X["short_name"].astype("category").cat.set_categories(ordered_symbols, ordered=True)
        return X