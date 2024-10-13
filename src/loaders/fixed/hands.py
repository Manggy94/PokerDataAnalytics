from sklearn.pipeline import Pipeline
from src.loaders.raw.hands import RawHandsLoader
from src.loaders.raw.ranks import RawRanksLoader
from src.loaders.raw.shapes import RawShapesLoader
from src.pipelines.hands import HandsPipeline


class HandsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_hands_loader", RawHandsLoader()),
            ("hands_pipeline", HandsPipeline(
                ranks=RawRanksLoader().transform(None),
                shapes=RawShapesLoader().transform(None)
            ))
        ])