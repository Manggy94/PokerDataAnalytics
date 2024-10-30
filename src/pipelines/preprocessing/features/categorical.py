import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.preprocessing.card_rank_encoder import CardRankEncoder
from src.transformers.preprocessing.categorical_features_encoder import CategoricalFeaturesEncoder
from src.transformers.preprocessing.categorical_features_finder import CategoricalFeaturesFinder
from src.transformers.preprocessing.categorical_na_filler import CategoricalNaFiller
from src.transformers.preprocessing.features_isolator import FeaturesIsolator
from src.transformers.preprocessing.flop_dropper import FlopDropper
from src.transformers.preprocessing.hero_combo_encoder import HeroComboEncoder
from src.transformers.preprocessing.id_dropper import IdDropper
from src.transformers.preprocessing.player_name_dropper import PlayerNameDropper


class CategoricalFeaturesPipeline(Pipeline):
    def __init__(self):
        super().__init__([
            ('features_isolator', FeaturesIsolator()),
            ('id_dropper', IdDropper()),
            # ('flop_dropper', FlopDropper()),
            ("categorical_features_finder", CategoricalFeaturesFinder()),
            # ('categorical_na_filler', CategoricalNaFiller()),
            # ('card_rank_encoder', CardRankEncoder()),
            # ('player_name_dropper', PlayerNameDropper()),
            ('categorical_features_encoder', CategoricalFeaturesEncoder()),
        ])