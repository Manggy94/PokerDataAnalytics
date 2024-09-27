import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.hands.hands_ranks_merger import HandsRanksMerger
from src.transformers.hands.hands_shapes_merger import HandsShapesMerger


class HandsPipeline(Pipeline):

    def __init__(self, ranks: pd.DataFrame, shapes: pd.DataFrame):
        self.ranks = ranks
        self.shapes = shapes
        super().__init__(steps=[
            ("hands_ranks_merger", HandsRanksMerger(ranks)),
            ("hands_shapes_merger", HandsShapesMerger(shapes)),
        ])