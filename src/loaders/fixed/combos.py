from sklearn.pipeline import Pipeline
from src.loaders.fixed.cards import CardsLoader
from src.loaders.fixed.hands import HandsLoader
from src.loaders.raw.combos import RawCombosLoader
from src.pipelines.components.combos import CombosPipeline


class CombosLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_combos_loader", RawCombosLoader()),
            ("combos_pipeline", CombosPipeline(
                cards=CardsLoader().transform(None),
                hands=HandsLoader().transform(None)
            ))
        ])