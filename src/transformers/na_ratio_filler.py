import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NaRatioFiller(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keywords = ["ratio"]
        self.ratio_cols = None

    def fit(self, X: pd.DataFrame, y=None):
        self.ratio_cols = [col for col in X.columns if any(keyword in col for keyword in self.keywords)]
        return self

    def transform(self, X: pd.DataFrame):
        for col in self.ratio_cols:
            X[col] = X[col].fillna(0)
        return X