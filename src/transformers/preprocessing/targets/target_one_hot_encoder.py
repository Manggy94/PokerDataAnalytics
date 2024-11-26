import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetOneHotEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return pd.get_dummies(X)

    def inverse_transform(self, X: pd.DataFrame):
        return X.idxmax(axis=1)
