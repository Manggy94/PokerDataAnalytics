from sklearn.pipeline import Pipeline

from src.loaders.dynamic.actions_sequences import ActionsSequencesLoader
from src.loaders.fixed.action_moves import ActionMovesLoader
from src.loaders.raw.player_hand_stats.flop import RawFlopPlayerHandStatsLoader
from src.pipelines.components.player_hand_stats.street import StreetPlayerHandStatsPipeline


class FlopPlayerHandStatsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_flop_player_hand_stats_loader", RawFlopPlayerHandStatsLoader()),
            ("street_player_hand_stats_pipeline", StreetPlayerHandStatsPipeline(
                action_moves=ActionMovesLoader().fit_transform(None),
                sequences=ActionsSequencesLoader().fit_transform(None)
            ))
        ])