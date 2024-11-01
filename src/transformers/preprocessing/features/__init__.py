import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.pipelines.preprocessing.features.categorical import CategoricalFeaturesPipeline
from src.pipelines.preprocessing.features.numerical import NumericalFeaturesPipeline


class GlobalFeaturesPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.categorical_features_pipeline = None
        self.numerical_features_pipeline = None

    def fit(self, X, y=None):
        self.categorical_features_pipeline = CategoricalFeaturesPipeline()
        self.numerical_features_pipeline = NumericalFeaturesPipeline()
        self.categorical_features_pipeline.fit(X)
        self.numerical_features_pipeline.fit(X)
        return self

    def transform(self, X):
        X_cat = self.categorical_features_pipeline.transform(X)
        X_num = self.numerical_features_pipeline.transform(X)
        return pd.concat([X_num, X_cat], axis=1).fillna(-1)