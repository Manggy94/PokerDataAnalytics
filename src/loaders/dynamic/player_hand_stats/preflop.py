from sklearn.pipeline import Pipeline
from src.loaders.raw.action_moves import RawActionMovesLoader
from src.loaders.raw.actions_sequences import RawActionsSequencesLoader
from src.loaders.raw.player_hand_stats.preflop import RawPreflopPlayerHandStatsLoader
from src.pipelines.player_hand_stats.street import StreetPlayerHandStatsPipeline


class PreflopPlayerHandStatsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_preflop_player_hand_stats_loader", RawPreflopPlayerHandStatsLoader()),
            ("street_player_hand_stats_pipeline", StreetPlayerHandStatsPipeline(
                action_moves=RawActionMovesLoader().fit_transform(None),
                sequences=RawActionsSequencesLoader().fit_transform(None)
            ))
        ])