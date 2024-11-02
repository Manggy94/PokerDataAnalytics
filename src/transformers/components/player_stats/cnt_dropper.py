import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CountDropper(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        cnt_cols = [col for col in X.columns if 'cnt_' in col]
        return X.drop(columns=cnt_cols)