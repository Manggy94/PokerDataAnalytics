import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.mappers.card_to_combos import card_to_combos_matrix
from src.mappers.combos.categorical.second_card import second_card_matrix


class TargetCorrector(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        forbidden_combos_matrix = 1 - card_to_combos_matrix()
        cards_index = forbidden_combos_matrix.index
        cards_columns = [col for col in X.columns if (col.endswith("_card") and not "hero" in col)]
        hero_data = X[["flag_is_hero", "hero_combo_first_card", "hero_combo_second_card"]]
        hero_dummies = pd.DataFrame(columns=cards_index, index=X.index).fillna(0).astype(int)
        first_card_dummies = pd.get_dummies(hero_data["hero_combo_first_card"], columns=cards_index)
        second_card_dummies = pd.get_dummies(hero_data["hero_combo_second_card"], columns=cards_index)
        hero_dummies = hero_dummies.add(first_card_dummies, fill_value=0).astype(int)
        hero_dummies = hero_dummies.add(second_card_dummies, fill_value=0).astype(int)
        board_dummies_list = [pd.get_dummies(X[col], columns=cards_index) for col in cards_columns]
        board_dummies = sum(board_dummies_list).fillna(0).astype(int)
        combined_dummies = board_dummies + hero_dummies
        max_val_vector = combined_dummies.sum(axis=1)
        product_vector = combined_dummies @ forbidden_combos_matrix
        y_corrector = (1*product_vector.ge(max_val_vector, axis=0))
        return y_corrector
