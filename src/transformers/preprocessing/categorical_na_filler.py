import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalNaFiller(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        categorical_columns = X.select_dtypes(include=['category']).columns
        for col in categorical_columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].cat.add_categories("None")
                X[col].fillna("None", inplace=True)
        return X