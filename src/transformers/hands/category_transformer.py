import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.helpers import map_int_dtype

class CategoryTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X["id"] = map_int_dtype(X["id"])
            obj_cols = X.select_dtypes(include=["object"]).columns
            X[obj_cols] = X[obj_cols].astype("category")
            X["rank_difference"] = X["rank_difference"].astype("uint8")
            return X