import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalFeaturesFinder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.categorical_features = None


    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.select_dtypes(include=['category'])