import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandsShapesMerger(BaseEstimator, TransformerMixin):

        def __init__(self, shapes: pd.DataFrame):
            self.columns_to_drop = ["short_name", "symbol"]
            self.shapes = shapes.drop(columns=self.columns_to_drop)

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            return X\
                .merge(self.shapes, how="left", left_on="shape", right_on="id", suffixes=("", "_shape"))\
                .drop(columns=["id_shape", "shape"])\
                .rename(columns={"name": "shape"})