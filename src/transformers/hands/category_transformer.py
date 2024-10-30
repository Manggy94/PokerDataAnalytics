import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X["id"] = X["id"].astype("uint8")
            X["short_name"] = X["short_name"].astype("category")
            X["rank_difference"] = X["rank_difference"].astype("uint8")
            return X