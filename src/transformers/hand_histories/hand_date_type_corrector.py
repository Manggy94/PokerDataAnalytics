import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandDateTypeCorrector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X["hand_date"] = pd.to_datetime(X["hand_date"])
        return X.sort_values(by="hand_date")
