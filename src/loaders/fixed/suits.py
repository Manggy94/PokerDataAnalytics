from sklearn.pipeline import Pipeline
from src.loaders.raw.suits import RawSuitsLoader
from src.transformers.suits.category_transformer import CategoryTransformer


class SuitsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_suits_loader", RawSuitsLoader()),
            ("category_transformer", CategoryTransformer()),
        ])