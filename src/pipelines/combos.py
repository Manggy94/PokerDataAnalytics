import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.combos.combos_cards_merger import CombosCardsMerger
from src.transformers.combos.combos_hands_merger import CombosHandsMerger


class CombosPipeline(Pipeline):

    def __init__(self, cards: pd.DataFrame, hands: pd.DataFrame):
        self.cards = cards
        self.hands = hands
        super().__init__(steps=[
            ("combos_hands_merger", CombosHandsMerger(hands)),
            ("combos_cards_merger", CombosCardsMerger(cards)),

        ])