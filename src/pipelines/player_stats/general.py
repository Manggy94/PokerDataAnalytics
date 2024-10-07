import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_stats.int_converter import IntConverter
from src.transformers.player_stats.player_stats_general_merger import PlayerStatsGeneralMerger


class GeneralPlayerStatsPipeline(Pipeline):
    def __init__(self, general_player_stats: pd.DataFrame):
        self.general_player_stats = general_player_stats
        super().__init__(steps=[
            ("general_stats_merger", PlayerStatsGeneralMerger(general_player_stats)),
            ("int_converter", IntConverter())
        ])