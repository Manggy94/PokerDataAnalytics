import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class CategoricalFeaturesEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        cat_columns = X.select_dtypes(include=['category']).columns
        X[cat_columns] = OrdinalEncoder().fit_transform(X[cat_columns]).astype("int16")
        return X
