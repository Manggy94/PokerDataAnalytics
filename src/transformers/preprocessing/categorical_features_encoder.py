import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

class CategoricalFeaturesEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        cat_data = X.select_dtypes(include=['category'])
        unique_values = cat_data.describe().T.unique
        ordinal_columns = unique_values[unique_values >=10].index.values
        one_hot_columns = unique_values[unique_values < 10].index.values
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        one_hot_encoder_data = one_hot_encoder.fit_transform(X[one_hot_columns])
        one_hot_encoder_col_names = one_hot_encoder.get_feature_names_out()
        one_hot_encoder_df = pd.DataFrame(one_hot_encoder_data, columns=one_hot_encoder_col_names, index=X.index)
        print(one_hot_encoder_df.shape)
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[ordinal_columns] = oe.fit_transform(X[ordinal_columns])
        print(X[ordinal_columns].shape)
        X = X.join(one_hot_encoder_df).drop(columns=one_hot_columns)
        return X
