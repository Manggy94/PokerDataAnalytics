from sklearn.pipeline import Pipeline
from src.transformers.preprocessing.targets.column_selector import ColumnSelector
from src.transformers.preprocessing.targets.target_encoder import TargetEncoder

class TargetPreprocessor(Pipeline):
    def __init__(self, target_column: str):
        self.target_column = target_column
        super().__init__([
            ('column_selector', ColumnSelector(target_column)),
            ('target_encoder', TargetEncoder())
            ])

    def inverse_transform(self, X=None, *, Xt=None, **params):
        self.steps[-1][1].encoder.inverse_transform(X, Xt, **params)
