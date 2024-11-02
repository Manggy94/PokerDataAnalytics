from sklearn.pipeline import Pipeline
from src.loaders.raw.tour_types import RawTourTypesLoader
from src.transformers.utils.category_transformer import CategoryTransformer


class TourTypesLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_tour_types_loader", RawTourTypesLoader()),
            ("category_transformer", CategoryTransformer()),
        ])