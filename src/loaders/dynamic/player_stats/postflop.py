from sklearn.pipeline import Pipeline
from src.loaders.raw.player_stats import RawPlayerStatsLoader
from src.loaders.raw.player_stats.flop import RawFlopPlayerStatsLoader
from src.loaders.raw.player_stats.turn import RawTurnPlayerStatsLoader
from src.loaders.raw.player_stats.river import RawRiverPlayerStatsLoader
from src.pipelines.components.player_stats.postflop import PostflopPlayerStatsPipeline


class PostflopPlayerStatsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ('raw_player_stats_loader', RawPlayerStatsLoader()),
            ('postflop_player_stats_pipeline', PostflopPlayerStatsPipeline(
                raw_flop_player_stats=RawFlopPlayerStatsLoader().fit_transform(None),
                raw_turn_player_stats=RawTurnPlayerStatsLoader().fit_transform(None),
                raw_river_player_stats=RawRiverPlayerStatsLoader().fit_transform(None))
                )
        ])