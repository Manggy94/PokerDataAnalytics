import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            obj_cols = X.select_dtypes(include=["object"]).columns
            X["id"] = X["id"].astype("uint8")
            X[obj_cols] = X[obj_cols].astype("category")
            return X