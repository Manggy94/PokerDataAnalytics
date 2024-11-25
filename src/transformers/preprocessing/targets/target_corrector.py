import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.mappers.card_to_combos import card_to_combos_matrix


class TargetCorrector(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        forbidden_combos_matrix = 1 - card_to_combos_matrix()
        cards_index = forbidden_combos_matrix.index
        cards_columns = [col for col in X.columns if col.endswith("_card")]
        dummies = [
            pd.get_dummies(X[col], columns=cards_index)
            for col in cards_columns
        ]
        combined_dummies = sum(dummies).fillna(0).astype(int)
        max_val_vector = combined_dummies.sum(axis=1)
        product_vector = combined_dummies @ forbidden_combos_matrix
        y_corrector = (1*product_vector.ge(max_val_vector, axis=0))
        return y_corrector
