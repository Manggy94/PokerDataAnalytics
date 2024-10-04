import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandIdDropper(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.drop_duplicates(subset=['hand_id'])
        X.drop(columns=['hand_id'], inplace=True)
        return X