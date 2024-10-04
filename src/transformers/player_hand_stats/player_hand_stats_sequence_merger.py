import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HandStatsSequenceMerger(BaseEstimator, TransformerMixin):

    def __init__(self, sequences: pd.DataFrame):
        self.sequences = sequences

    def fit(self, X, y=None):
        self.sequence_columns = [col for col in X.columns if "sequence" in col]
        return self

    def transform(self, X: pd.DataFrame):
        for col in self.sequence_columns:
            X = X\
                .merge(self.sequences, how="left", left_on=col, right_on="id", suffixes=("", f"_{col}"))\
                .drop(columns=[col, f"id_{col}"])\
                .rename(columns={"symbol": col})
        return X