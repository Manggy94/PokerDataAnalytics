import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.helpers import map_int_dtype

class RawDtypesCorrector(BaseEstimator, TransformerMixin):

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X["combo"] = map_int_dtype(X.combo)
            X["position"] = map_int_dtype(X.position)
            return X