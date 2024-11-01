import pandas as pd
from sklearn.pipeline import Pipeline

from src.transformers.preprocessing.bool_to_int_converter import BoolToIntConverter
from src.transformers.preprocessing.features.features_isolator import FeaturesIsolator
from src.transformers.preprocessing.id_dropper import IdDropper
from src.transformers.preprocessing.numerical_features_finder import NumericalFeaturesFinder


class NumericalFeaturesPipeline(Pipeline):
    def __init__(self):
        super().__init__([
            ('features_isolator', FeaturesIsolator()),
            ('id_dropper', IdDropper()),
            ('bool_to_int_converter', BoolToIntConverter()),
            ("numerical_features_finder", NumericalFeaturesFinder()),
        ])