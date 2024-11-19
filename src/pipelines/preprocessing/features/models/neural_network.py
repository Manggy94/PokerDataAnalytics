from sklearn.pipeline import Pipeline
from src.transformers.preprocessing.features import GlobalFeaturesPreprocessor
from src.transformers.preprocessing.features.numerical.min_max_scaler import MinMaxScaler



class NeuralNetworkFeaturesPreprocessor(Pipeline):
    def __init__(self):
        super().__init__(
            steps=[
                ('global_features_preprocessor', GlobalFeaturesPreprocessor()),
                ('min_max_scaler', MinMaxScaler())

            ]
        )
