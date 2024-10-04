import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ObjectsCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.objects = None

    def fit(self, X: pd.DataFrame, y=None):
        self.objects = X.select_dtypes(include=["object"]).columns
        return self

    def transform(self, X: pd.DataFrame):
        for obj in self.objects:
            X[obj] = X[obj].astype("category")
        return X