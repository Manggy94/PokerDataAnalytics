import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BBNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.value_columns = None
        self.keywords = ["stack", "amount", "chips"]

    def fit(self, X: pd.DataFrame, y=None):
        self.value_columns = [col for col in X.columns if any([keyword in col for keyword in self.keywords])]
        return self

    def transform(self, X: pd.DataFrame):
        for col in self.value_columns:
            X = X.copy()
            X[f"{col}_bb"] = X[col] / X["level_bb"]
        return X