import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.cards.cards_ranks_merger import CardsRanksMerger
from src.transformers.cards.cards_suits_merger import CardsSuitsMerger
from src.transformers.cards.category_transformer import CategoryTransformer


class CardsPipeline(Pipeline):
    """A pipeline to return complete cards data."""
    def __init__(self, ranks: pd.DataFrame = None, suits: pd.DataFrame = None):
        self.ranks = ranks
        self.suits = suits
        super().__init__(steps=[
            ("cards_ranks_merger", CardsRanksMerger(ranks)),
            ("cards_suits_merger", CardsSuitsMerger(suits)),
            ("category_transformer", CategoryTransformer()),

        ])