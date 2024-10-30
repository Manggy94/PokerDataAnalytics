import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X["id"] = X["id"].astype("int8")
            X["name"] = X["name"].astype("category")
            X["symbol"] = X["symbol"].astype("category")
            X["short_name"] = X["short_name"].astype("category")
            return X