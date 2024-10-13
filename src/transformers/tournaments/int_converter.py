import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IntConverter(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.int_8_columns = None
        self.int_32_columns = None
        self.int_8_keywords = ["nb_"]
        self.int_32_keywords = ["total_players", "final_position", "starting_stack"]

    def fit(self, X: pd.DataFrame, y=None):
        self.int_8_columns = [col for col in X.columns if any(keyword in col for keyword in self.int_8_keywords)]
        self.int_32_columns = [col for col in X.columns if any(keyword in col for keyword in self.int_32_keywords)]
        return self

    def transform(self, X: pd.DataFrame):
        X[self.int_8_columns] = X[self.int_8_columns].astype("Int8")
        X[self.int_32_columns] = X[self.int_32_columns].astype("Int32")
        return X