import pandas as pd
from sklearn.pipeline import Pipeline

from src.transformers.flops.category_transformer import CategoryTransformer
from src.transformers.flops.flops_cards_merger import FlopsCardsMerger


class FlopsPipeline(Pipeline):
    def __init__(self, cards: pd.DataFrame):
        self.cards = cards
        super().__init__(steps=[
            ('category_transformer', CategoryTransformer()),
            ("flops_cards_merger", FlopsCardsMerger(cards))
        ])
