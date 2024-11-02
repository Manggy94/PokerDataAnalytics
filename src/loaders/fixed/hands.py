from sklearn.pipeline import Pipeline

from src.loaders.fixed.ranks import RanksLoader
from src.loaders.fixed.shapes import ShapesLoader
from src.loaders.raw.hands import RawHandsLoader
from src.pipelines.components.hands import HandsPipeline


class HandsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_hands_loader", RawHandsLoader()),
            ("hands_pipeline", HandsPipeline(
                ranks=RanksLoader().transform(None),
                shapes=ShapesLoader().transform(None)
            ))
        ])