import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.combos.combos_cards_merger import CombosCardsMerger


class CombosPipeline(Pipeline):

    def __init__(self, cards: pd.DataFrame, hands: pd.DataFrame):
        self.cards = cards
        self.hands = hands
        super().__init__(steps=[
            ("combos_cards_merger", CombosCardsMerger(cards))
        ])