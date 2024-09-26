import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.cards.cards_ranks_merger import CardsRanksMerger
from src.transformers.cards.cards_suits_merger import CardsSuitsMerger


class CardsPipeline(Pipeline):

        def __init__(self, ranks: pd.DataFrame, suits: pd.DataFrame):
            self.ranks = ranks
            self.suits = suits
            super().__init__(steps=[
                ("cards_ranks_merger", CardsRanksMerger(ranks)),
                ("cards_suits_merger", CardsSuitsMerger(suits))
            ])