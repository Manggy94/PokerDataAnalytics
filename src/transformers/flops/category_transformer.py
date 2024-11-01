import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.helpers import map_int_dtype


class CategoryTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X["id"] = map_int_dtype(X["id"])
            X["short_name"] = X["short_name"].astype("category")
            X["symbol"] = X["symbol"].astype("category")
            X["min_distance"] = map_int_dtype(X["min_distance"])
            X["max_distance"] = map_int_dtype(X["max_distance"])
            return X