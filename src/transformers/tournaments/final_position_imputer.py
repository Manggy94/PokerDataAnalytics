import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FinalPositionImputer(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X["final_position"] = X["final_position"].fillna(X["total_players"]).astype("int32")
        return X