import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FloatConverter(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        float_cols = X.select_dtypes(include=['float64']).columns
        X[float_cols] = X[float_cols].astype('float32')
        return X