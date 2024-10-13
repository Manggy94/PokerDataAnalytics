from sklearn.pipeline import Pipeline
from src.loaders.fixed.combos import CombosLoader
from src.loaders.fixed.positions import PositionsLoader
from src.loaders.raw.action_moves import RawActionMovesLoader
from src.loaders.raw.player_hand_stats.general import RawGeneralPlayerHandStatsLoader
from src.loaders.raw.streets import RawStreetsLoader
from src.pipelines.player_hand_stats.general import GeneralPlayerHandStatsPipeline


class GeneralPlayerHandStatsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_general_player_hand_stats_loader", RawGeneralPlayerHandStatsLoader()),
            ("general_player_hand_stats_pipeline", GeneralPlayerHandStatsPipeline(
                action_moves=RawActionMovesLoader().fit_transform(None),
                combos=CombosLoader().fit_transform(None),
                positions=PositionsLoader().fit_transform(None),
                streets=RawStreetsLoader().fit_transform(None)
            ))
        ])