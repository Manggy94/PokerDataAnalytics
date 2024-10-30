import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config.settings import ANALYTICS_DATA_DIR


class RawTourTypesLoader(BaseEstimator, TransformerMixin):
    """ A transformer to load the raw tour types data from csv """
    def __init__(self):
        self.ANALYTICS_DATA_DIR = ANALYTICS_DATA_DIR

    def fit(self, X = None, y = None):
        return self

    def transform(self, X=None):
        return pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/tour_types.csv', index_col=0).sort_values("id")