from sklearn.pipeline import Pipeline
from src.loaders.raw.action_moves import RawActionMovesLoader
from src.transformers.utils.category_transformer import CategoryTransformer


class ActionMovesLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_action_moves_loader", RawActionMovesLoader()),
            ('category_transformer', CategoryTransformer())
        ])