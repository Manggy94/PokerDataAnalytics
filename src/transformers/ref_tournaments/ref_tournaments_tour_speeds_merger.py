import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RefTournamentsTourSpeedsMerger(BaseEstimator, TransformerMixin):
    def __init__(self, tour_speeds: pd.DataFrame):
        self.tour_speeds = tour_speeds

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X\
            .merge(self.tour_speeds, how='left', left_on='speed', right_on='id', suffixes=('', '_speed'))\
            .drop(columns=['id_speed', 'speed'])