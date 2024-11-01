from sklearn.pipeline import Pipeline

from src.loaders.dynamic.actions_sequences import ActionsSequencesLoader
from src.loaders.fixed.action_moves import ActionMovesLoader
from src.loaders.raw.player_hand_stats.turn import RawTurnPlayerHandStatsLoader
from src.pipelines.player_hand_stats.street import StreetPlayerHandStatsPipeline


class TurnPlayerHandStatsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_turn_player_hand_stats_loader", RawTurnPlayerHandStatsLoader()),
            ("street_player_hand_stats_pipeline", StreetPlayerHandStatsPipeline(
                action_moves=ActionMovesLoader().fit_transform(None),
                sequences=ActionsSequencesLoader().fit_transform(None)
            ))
        ])