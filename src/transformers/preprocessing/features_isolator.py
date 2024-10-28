import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeaturesIsolator(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        y_columns = [col for col in X.columns if "player_combo" in col]
        X = X.drop(columns=[y_columns])
        return X