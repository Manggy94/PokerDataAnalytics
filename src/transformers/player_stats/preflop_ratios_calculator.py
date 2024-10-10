import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config.settings import ANALYTICS_DATA_DIR


class PreflopRatiosCalculator(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.stats_df = pd.read_csv(f"{ANALYTICS_DATA_DIR}/player_stats_tables/preflop.csv")
        self.num_cols = self.stats_df.num_col
        self.denum_cols = self.stats_df.denum_col
        self.stat_names = self.stats_df.stat_name


    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for num_col, denum_col, stat_name in zip(self.num_cols, self.denum_cols, self.stat_names):
            X[stat_name] = (X[num_col] / X[denum_col]).fillna(0).replace(np.inf, 1).astype("float16")
        return X