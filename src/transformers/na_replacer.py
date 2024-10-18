import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NaReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, keywords: list, source_column: str):
        self.keywords = keywords
        self.source_column = source_column
        self.columns_to_replace = None

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_to_replace = [col for col in X.columns if any(keyword in col for keyword in self.keywords)]
        return self

    def transform(self, X: pd.DataFrame):
        for col in self.columns_to_replace:
            X[col] = np.where(X[self.source_column].isna(), np.nan, X[col])
        return X