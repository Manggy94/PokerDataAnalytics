import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class FeaturesScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        X = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X: pd.DataFrame):
        X = pd.DataFrame(self.scaler.inverse_transform(X), columns=X.columns, index=X.index)
        return X