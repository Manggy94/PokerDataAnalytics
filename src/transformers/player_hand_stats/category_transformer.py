import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoryTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.ordered_rank_symbols = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        self.rank_keywords = ["first_rank", "second_rank", "card_rank"]
        self.rank_cols = None

    def fit(self, X: pd.DataFrame, y=None):
        self.rank_cols = [col for col in X.columns if any(keyword in col for keyword in self.rank_keywords)]
        self.obj_cols = X.select_dtypes(include="object").columns
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.rank_cols:
            X[col] = X[col].astype("category").cat.set_categories(self.ordered_rank_symbols, ordered=True)
        for col in self.obj_cols:
            X[col] = X[col].astype("category")
        return X