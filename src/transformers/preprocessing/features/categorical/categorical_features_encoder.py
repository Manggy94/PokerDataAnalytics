import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

class CategoricalFeaturesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.ordinal_columns = self.one_hot_columns = None

    def fit(self, X: pd.DataFrame, y=None):
        cat_data = X.select_dtypes(include=['category'])
        unique_values = cat_data.describe().T.unique
        self.ordinal_columns = unique_values[unique_values >= 10].index.values
        self.one_hot_columns = unique_values[unique_values < 10].index.values
        self.one_hot_encoder.fit(X[self.one_hot_columns])
        self.ordinal_encoder.fit(X[self.ordinal_columns])
        return self

    def transform(self, X: pd.DataFrame):
        one_hot_df = pd.DataFrame(
            self.one_hot_encoder.transform(X[self.one_hot_columns]),
            columns=self.one_hot_encoder.get_feature_names_out(),
            index=X.index
        )
        X[self.ordinal_columns] = self.ordinal_encoder.transform(X[self.ordinal_columns])
        return X.join(one_hot_df).drop(columns=self.one_hot_columns)