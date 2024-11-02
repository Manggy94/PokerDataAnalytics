from sklearn.pipeline import Pipeline
from src.loaders.raw.player_stats import RawPlayerStatsLoader
from src.loaders.raw.player_stats.general import RawGeneralPlayerStatsLoader
from src.pipelines.components.player_stats.general import GeneralPlayerStatsPipeline



class GeneralPlayerStatsLoader(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("raw_player_stats_loader", RawPlayerStatsLoader()),
            ("general_player_stats_pipeline", GeneralPlayerStatsPipeline(
                raw_general_player_stats=RawGeneralPlayerStatsLoader().fit_transform(None),
            ))
        ])