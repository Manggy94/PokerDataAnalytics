import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NumericalNaFiller(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        numerical_columns = X.select_dtypes(include=['number']).columns
        for col in numerical_columns:
            if X[col].isna().sum() > 0:
                X[col].fillna(-1, inplace=True)
        return X