import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TournamentRefTournamentsMerger(BaseEstimator, TransformerMixin):
    def __init__(self, ref_tournaments: pd.DataFrame):
        self.ref_tournaments = ref_tournaments

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X):
        return X\
            .merge(self.ref_tournaments, how='left', left_on='ref_tournament', right_on='ref_tournament_id',
                   suffixes=('', '_ref'))\
            .drop(columns=['ref_tournament', 'ref_tournament_id'])