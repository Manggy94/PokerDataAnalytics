import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X["id"] = X["id"].astype("uint16")
            X["short_name"] = X["short_name"].astype("category")
            X["symbol"] = X["symbol"].astype("category")
            X["min_distance"] = X["min_distance"].astype("uint8")
            X["max_distance"] = X["max_distance"].astype("uint8")
            return X