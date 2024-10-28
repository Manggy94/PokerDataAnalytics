from sklearn.pipeline import Pipeline
from src.transformers.preprocessing.categorical_na_filler import CategoricalNaFiller
from src.transformers.preprocessing.datetime_dropper import DateTimeDropper
from src.transformers.preprocessing.features_isolator import FeaturesIsolator
from src.transformers.preprocessing.id_dropper import IdDropper
from src.transformers.preprocessing.numerical_na_filler import NumericalNaFiller
from src.transformers.preprocessing.objects_dropper import ObjectsDropper

class ClassificationPreprocessor(Pipeline):
    def __init__(self):
        super().__init__(
            steps=[
                ('id_dropper', IdDropper()),
                ('objects_dropper', ObjectsDropper()),
                ('datetime_dropper', DateTimeDropper()),
                ('categorical_na_filler', CategoricalNaFiller()),
                ('numerical_na_filler', NumericalNaFiller()),
                ('features_isolator', FeaturesIsolator())
            ]
        )
