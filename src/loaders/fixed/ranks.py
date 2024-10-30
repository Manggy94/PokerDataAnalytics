from sklearn.pipeline import Pipeline
from src.loaders.raw.ranks import RawRanksLoader
from src.transformers.ranks.category_transformer import CategoryTransformer


class RanksLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_ranks_loader", RawRanksLoader()),
            ("category_transformer", CategoryTransformer()),
        ])