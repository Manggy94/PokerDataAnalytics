import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FloatConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.float_columns = None
        self.float_keywords = ["amount", "ratio", "_bb", "_sb", "_ante", "bounty", "stack", "chips", "buy_in"]

    def fit(self, X, y=None):
        self.float_columns = [col for col in X.columns if any(keyword in col for keyword in self.float_keywords)]
        return self

    def transform(self, X: pd.DataFrame):
        X[self.float_columns] = X[self.float_columns].astype("float32")
        return X