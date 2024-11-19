from sklearn.pipeline import Pipeline
from src.transformers.preprocessing.targets.column_selector import ColumnSelector
from src.transformers.preprocessing.targets.target_one_hot_encoder import TargetOneHotEncoder

class NeuralNetworkTargetPreprocessor(Pipeline):
    def __init__(self, target_column: str):
        self.target_column = target_column
        super().__init__([
            ('column_selector', ColumnSelector(target_column)),
            ('target_encoder', TargetOneHotEncoder())
            ])

    def inverse_transform(self, X):
        self.steps[-1][1].encoder.inverse_transform(X)
