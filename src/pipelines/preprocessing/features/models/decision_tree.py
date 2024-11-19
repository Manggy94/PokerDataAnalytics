from sklearn.pipeline import Pipeline
from src.transformers.preprocessing.features import GlobalFeaturesPreprocessor


class RandomForestFeaturesPreprocessor(Pipeline):
    def __init__(self):
        super().__init__(
            steps=[
                ('global_features_preprocessor', GlobalFeaturesPreprocessor())
            ]
        )
