from sklearn.pipeline import Pipeline
from src.loaders.raw.streets import RawStreetsLoader
from src.transformers.streets.category_transformer import CategoryTransformer


class StreetsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_streets_loader", RawStreetsLoader()),
            ("category_transformer", CategoryTransformer()),
        ])