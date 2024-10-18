from sklearn.pipeline import Pipeline
from src.loaders.dynamic.tournaments import TournamentsLoader
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
                cards=CardsLoader().fit_transform(None),
                combos=CombosLoader().fit_transform(None),
                flops=FlopsLoader().fit_transform(None),
                levels=RawLevelsLoader().fit_transform(None),
                tournaments=TournamentsLoader().fit_transform(None)
            ))
        ])