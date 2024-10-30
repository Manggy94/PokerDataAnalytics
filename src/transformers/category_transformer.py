import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.helpers import map_id_dtype

class CategoryTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X["id"] = X["id"].astype(map_id_dtype(X))
            obj_cols = X.select_dtypes(include=["object"]).columns
            X[obj_cols] = X[obj_cols].astype("category")
            return X