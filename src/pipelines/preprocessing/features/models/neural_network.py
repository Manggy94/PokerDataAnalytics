import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.preprocessing.features import GlobalFeaturesPreprocessor
from src.transformers.preprocessing.features.numerical.min_max_scaler import FeaturesScaler



class NeuralNetworkFeaturesPreprocessor(Pipeline):
    def __init__(self):
        super().__init__(
            steps=[
                ('global_features_preprocessor', GlobalFeaturesPreprocessor()),
                ('min_max_scaler', FeaturesScaler())
            ]
        )

    def retrieve_known_cards(self, X):
        X_unscaled = self.named_steps['min_max_scaler'].inverse_transform(X=X)
        cards_encoder = self.named_steps['global_features_preprocessor'] \
            .categorical_features_pipeline \
            .named_steps['categorical_features_encoder'] \
            .ordinal_encoder
        ordinal_columns = cards_encoder.feature_names_in_
        X_ordinal = X_unscaled[ordinal_columns].astype(int)
        ordinal_data_retrieved = cards_encoder.inverse_transform(X_ordinal)
        X_ordinal_retrieved = pd.DataFrame(ordinal_data_retrieved, columns=ordinal_columns, index=X_ordinal.index)
        shown_cards_columns = [c for c in X_unscaled.columns if c.endswith("_card")]
        X_cards = X_ordinal_retrieved[shown_cards_columns].astype("category")
        X_cards = X_cards.join(X["flag_is_hero"])
        return X_cards
