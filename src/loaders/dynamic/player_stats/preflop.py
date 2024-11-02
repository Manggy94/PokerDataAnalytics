from sklearn.pipeline import Pipeline
from src.loaders.raw.player_stats import RawPlayerStatsLoader
from src.loaders.raw.player_stats.preflop import RawPreflopPlayerStatsLoader
from src.pipelines.components.player_stats.preflop import PreflopPlayerStatsPipeline


class PreflopPlayerStatsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ('raw_player_stats_loader', RawPlayerStatsLoader()),
            ('preflop_player_stats_pipeline', PreflopPlayerStatsPipeline(
                raw_preflop_player_stats=RawPreflopPlayerStatsLoader().fit_transform(None))
                )
        ])