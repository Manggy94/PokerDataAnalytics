import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandsRenamer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.rename(columns={c: f"hand_{c}" for c in X.columns})