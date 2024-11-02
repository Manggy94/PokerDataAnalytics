import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RefTournamentsBuyInsMerger(BaseEstimator, TransformerMixin):
    def __init__(self, buy_ins: pd.DataFrame):
        self.buy_ins = buy_ins

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X):
        return X\
            .merge(self.buy_ins, how='left', left_on='buy_in', right_on='id', suffixes=('', '_buy_in'))\
            .drop(columns=['id_buy_in', 'buy_in'])\
            .rename(columns={c: f"buy_in_{c}" for c in self.buy_ins.columns if c != 'id'})