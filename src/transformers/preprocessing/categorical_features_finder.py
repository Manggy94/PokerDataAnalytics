import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalFeaturesFinder(BaseEstimator, TransformerMixin):


    def fit(self, X: pd.DataFrame, y=None):
        self.cat_columns = X.select_dtypes(include=['category']).columns
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.cat_columns]