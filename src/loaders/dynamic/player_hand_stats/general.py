from sklearn.pipeline import Pipeline

from src.loaders.fixed.action_moves import ActionMovesLoader
from src.loaders.fixed.combos import CombosLoader
from src.loaders.fixed.positions import PositionsLoader
from src.loaders.fixed.streets import StreetsLoader
from src.loaders.raw.player_hand_stats.general import RawGeneralPlayerHandStatsLoader
from src.pipelines.player_hand_stats.general import GeneralPlayerHandStatsPipeline


class GeneralPlayerHandStatsLoader(Pipeline):
    def __init__(self):
        action_moves = ActionMovesLoader().fit_transform(None)
        combos = CombosLoader().fit_transform(None)
        positions = PositionsLoader().fit_transform(None)
        streets = StreetsLoader().fit_transform(None)
        super().__init__(steps=[
            ("raw_general_player_hand_stats_loader", RawGeneralPlayerHandStatsLoader()),
            ("general_player_hand_stats_pipeline", GeneralPlayerHandStatsPipeline(
                action_moves=action_moves,
                combos=combos,
                positions=positions,
                streets=streets
            ))
        ])