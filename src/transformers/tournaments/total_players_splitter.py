import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TotalPlayersSplitter(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.ranges = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
        self.labels = [f"{val[0]+1} - {val[1]}" for val in zip(self.ranges[:-1], self.ranges[1:])]

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X["total_players_range"] = pd.cut(X["total_players"], bins=self.ranges,
                                          labels=self.labels)
        X["total_players_range"] = pd.Categorical(X["total_players_range"], categories=self.labels, ordered=True)
        return X