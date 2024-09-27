import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FaceCoveringBetStreetMerger(BaseEstimator, TransformerMixin):

        def __init__(self, streets: pd.DataFrame):
            self.streets = streets

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            return X\
                .merge(self.streets, how="left", left_on="face_covering_bet_street", right_on="id", suffixes=("", "_street"))\
                .drop(columns=["id_street", "face_covering_bet_street", "name", "parsing_name", "short_name", "is_preflop"])\
                .rename(columns={"symbol": "face_covering_bet_street"})