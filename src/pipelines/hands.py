import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.hands.hands_ranks_merger import HandsRanksMerger

class HandsPipeline(Pipeline):

    def __init__(self, ranks: pd.DataFrame, suits: pd.DataFrame):
        self.ranks = ranks
        self.suits = suits
        super().__init__(steps=[
            ("hands_ranks_merger", HandsRanksMerger(ranks))
        ])