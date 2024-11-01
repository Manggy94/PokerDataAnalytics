import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BoolToIntConverter(BaseEstimator, TransformerMixin):


    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        bool_cols = X.select_dtypes(include=[np.bool8]).columns
        X[bool_cols] = X[bool_cols].astype(np.int8)
        return X