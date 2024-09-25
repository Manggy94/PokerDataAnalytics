import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RefTournamentsTourTypesMerger(BaseEstimator, TransformerMixin):
    def __init__(self, tour_types: pd.DataFrame):
        self.tour_types = tour_types

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.tour_types, how='left', left_on='tournament_type', right_on='id', suffixes=('', '_type'))\
            .drop(columns=['id_type', 'tournament_type'])