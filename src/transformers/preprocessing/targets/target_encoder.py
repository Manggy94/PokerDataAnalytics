import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class TargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X: pd.DataFrame, y=None):
        self.encoder.fit(X.cat.categories)
        return self

    def transform(self, X: pd.DataFrame):
        target = pd.Series(self.encoder.transform(X), name=X.name, index=X.index)
        return target
