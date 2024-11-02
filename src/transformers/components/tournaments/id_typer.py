import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IdTyper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X["tournament_id"] = X["tournament_id"].astype("str")
        return X