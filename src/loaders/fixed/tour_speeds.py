from sklearn.pipeline import Pipeline
from src.loaders.raw.tour_speeds import RawTourSpeedsLoader
from src.transformers.category_transformer import CategoryTransformer


class TourSpeedsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_tour_speeds_loader", RawTourSpeedsLoader()),
            ("category_transformer", CategoryTransformer()),
        ])
