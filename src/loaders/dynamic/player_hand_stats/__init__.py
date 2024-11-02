from sklearn.pipeline import Pipeline
from src.loaders.dynamic.hand_histories import HandHistoriesLoader
from src.loaders.dynamic.player_hand_stats.flop import FlopPlayerHandStatsLoader
from src.loaders.dynamic.player_hand_stats.general import GeneralPlayerHandStatsLoader
from src.loaders.dynamic.player_hand_stats.preflop import PreflopPlayerHandStatsLoader
from src.loaders.dynamic.player_hand_stats.river import RiverPlayerHandStatsLoader
from src.loaders.dynamic.player_hand_stats.turn import TurnPlayerHandStatsLoader
from src.loaders.dynamic.player_stats import PlayerStatsLoader
from src.loaders.raw.player_hand_stats import RawPlayerHandStatsLoader
from src.pipelines.components.player_hand_stats import PlayerHandStatsPipeline


class PlayerHandStatsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_player_hand_stats_loader", RawPlayerHandStatsLoader()),
            ("player_hand_stats_pipeline", PlayerHandStatsPipeline(
                hand_histories=HandHistoriesLoader().fit_transform(None),
                general_player_hand_stats=GeneralPlayerHandStatsLoader().fit_transform(None),
                preflop_player_hand_stats=PreflopPlayerHandStatsLoader().fit_transform(None),
                flop_player_hand_stats=FlopPlayerHandStatsLoader().fit_transform(None),
                turn_player_hand_stats=TurnPlayerHandStatsLoader().fit_transform(None),
                river_player_hand_stats=RiverPlayerHandStatsLoader().fit_transform(None),
                player_stats=PlayerStatsLoader().fit_transform(None)
                )
            )

        ])