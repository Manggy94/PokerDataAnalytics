import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Float64Reducer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        columns = X.select_dtypes(include=['float64']).columns
        X[columns] = X[columns].astype('float32')
        return X