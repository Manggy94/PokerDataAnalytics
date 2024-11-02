import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StartDateTypeCorrector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X["start_date"] = pd.to_datetime(X["start_date"])
        X = X.sort_values(by="start_date")
        return X