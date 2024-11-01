import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class CardRankEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keyword = "card_rank"

    def fit(self, X: pd.DataFrame, y=None):
        ranks_order = ["None", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        self.card_rank_columns = [col for col in X.columns if self.keyword in col]
        for col in self.card_rank_columns:
            X[col] = X[col].cat.set_categories(ranks_order, ordered=True)
        return self

    def transform(self, X: pd.DataFrame):
        X[self.card_rank_columns] = OrdinalEncoder().fit_transform(X[self.card_rank_columns]).astype("int8")
        return X