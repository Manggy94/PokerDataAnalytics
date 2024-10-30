import pandas as pd
from sklearn.pipeline import Pipeline

from src.transformers.category_transformer import CategoryTransformer
from src.transformers.combos.combos_cards_merger import CombosCardsMerger
from src.transformers.combos.combos_hands_merger import CombosHandsMerger


class CombosPipeline(Pipeline):

    def __init__(
            self,
            cards: pd.DataFrame = None,
            hands: pd.DataFrame = None
    ):
        self.cards = cards
        self.hands = hands
        super().__init__(steps=[
            ('category_transformer', CategoryTransformer()),
            ("combos_hands_merger", CombosHandsMerger(hands)),
            ("combos_cards_merger", CombosCardsMerger(cards)),

        ])