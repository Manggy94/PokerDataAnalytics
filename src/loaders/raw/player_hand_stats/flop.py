import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config.settings import ANALYTICS_DATA_DIR


class RawFlopPlayerHandStatsLoader(BaseEstimator, TransformerMixin):
    """ A transformer to load the raw flop player stats data from csv """
    def __init__(self):
        self.ANALYTICS_DATA_DIR = ANALYTICS_DATA_DIR

    def fit(self, X = None, y = None):
        return self

    def transform(self, X=None):
        return pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/flop_player_hand_stats.csv', index_col=0)