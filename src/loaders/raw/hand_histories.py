import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config.settings import ANALYTICS_DATA_DIR


class RawHandHistoriesLoader(BaseEstimator, TransformerMixin):
    """ A transformer to load the raw hand histories data from csv """
    def __init__(self):
        self.ANALYTICS_DATA_DIR = ANALYTICS_DATA_DIR

    def fit(self, X = None, y = None):
        return self

    def transform(self, X=None):
        hand_histories = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/hand_histories.csv', index_col=0)
        return hand_histories
