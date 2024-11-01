import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HeroComboEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        dummy_combos = pd.get_dummies(X["hero_combo"], prefix="hero_combo", sparse=True)
        dummy_hands = pd.get_dummies(X["hero_combo_hand"], prefix="hero_combo_hand", sparse=True)
        X = X.join(dummy_combos).join(dummy_hands).drop(columns=["hero_combo", "hero_combo_hand"])
        return X
