from sklearn.pipeline import Pipeline
from src.loaders.fixed.cards import CardsLoader
from src.loaders.fixed.combos import CombosLoader
from src.loaders.fixed.flops import FlopsLoader
from src.loaders.raw.hand_histories import RawHandHistoriesLoader
from src.loaders.raw.levels import RawLevelsLoader
from src.pipelines.hand_histories import HandHistoriesPipeline


class HandHistoriesLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_hand_histories_loader", RawHandHistoriesLoader()),
            ("hand_histories_pipeline", HandHistoriesPipeline(
                cards=CardsLoader().transform(None),
                combos=CombosLoader().transform(None),
                flops=FlopsLoader().transform(None),
                levels=RawLevelsLoader().transform(None)
            ))
        ])