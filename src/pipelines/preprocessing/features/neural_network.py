from sklearn.pipeline import Pipeline
from src.transformers.preprocessing.bool_to_int_converter import BoolToIntConverter
from src.transformers.preprocessing.categorical_features_encoder import CategoricalFeaturesEncoder
from src.transformers.preprocessing.categorical_na_filler import CategoricalNaFiller
from src.transformers.preprocessing.datetime_dropper import DateTimeDropper
from src.transformers.preprocessing.features_isolator import FeaturesIsolator
from src.transformers.preprocessing.id_dropper import IdDropper
from src.transformers.preprocessing.numerical_na_filler import NumericalNaFiller
from src.transformers.preprocessing.objects_dropper import ObjectsDropper
from src.transformers.preprocessing.player_name_dropper import PlayerNameDropper


class NeuralNetworkFeaturesPreprocessor(Pipeline):
    def __init__(self):
        super().__init__(
            steps=[
                ('features_isolator', FeaturesIsolator()),
                ('id_dropper', IdDropper()),
                ('player_name_dropper', PlayerNameDropper()),
                ('objects_dropper', ObjectsDropper()),
                ('datetime_dropper', DateTimeDropper()),
                ('categorical_na_filler', CategoricalNaFiller()),
                ('numerical_na_filler', NumericalNaFiller()),
                ('categorical_features_encoder', CategoricalFeaturesEncoder()),
                ('bool_to_int_converter', BoolToIntConverter()),

            ]
        )
