from sklearn.pipeline import Pipeline
from src.transformers.preprocessing.features.categorical.categorical_features_encoder import CategoricalFeaturesEncoder
from src.transformers.preprocessing.features.categorical.categorical_features_finder import CategoricalFeaturesFinder
from src.transformers.preprocessing.utils.features_isolator import FeaturesIsolator
from src.transformers.preprocessing.utils.id_dropper import IdDropper


class CategoricalFeaturesPipeline(Pipeline):
    def __init__(self):
        super().__init__([
            ('features_isolator', FeaturesIsolator()),
            ('id_dropper', IdDropper()),
            ("categorical_features_finder", CategoricalFeaturesFinder()),
            ('categorical_features_encoder', CategoricalFeaturesEncoder()),
        ])