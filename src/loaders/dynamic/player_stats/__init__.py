from sklearn.pipeline import Pipeline
from src.loaders.raw.players import RawPlayersLoader
from src.loaders.raw.player_stats import RawPlayerStatsLoader
from src.loaders.raw.player_stats.general import RawGeneralPlayerStatsLoader
from src.loaders.raw.player_stats.preflop import RawPreflopPlayerStatsLoader
from src.loaders.raw.player_stats.flop import RawFlopPlayerStatsLoader
from src.loaders.raw.player_stats.turn import RawTurnPlayerStatsLoader
from src.loaders.raw.player_stats.river import RawRiverPlayerStatsLoader
from src.pipelines.player_stats import PlayerStatsPipeline
from src.transformers.player_names_merger import PlayerNamesMerger


class PlayerStatsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ('raw_player_stats_loader', RawPlayerStatsLoader()),
            ('player_names_merger', PlayerNamesMerger(
                player_names=RawPlayersLoader().fit_transform(None),
                player_id_col='player')),
            ('player_stats_pipeline', PlayerStatsPipeline(
                raw_general_player_stats=RawGeneralPlayerStatsLoader().fit_transform(None),
                raw_preflop_player_stats=RawPreflopPlayerStatsLoader().fit_transform(None),
                raw_flop_player_stats=RawFlopPlayerStatsLoader().fit_transform(None),
                raw_turn_player_stats=RawTurnPlayerStatsLoader().fit_transform(None),
                raw_river_player_stats=RawRiverPlayerStatsLoader().fit_transform(None)

            ))
        ])