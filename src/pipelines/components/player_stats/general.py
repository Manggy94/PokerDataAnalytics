import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.utils.float_converter import FloatConverter
from src.transformers.components.player_stats.general_ratios_calculator import GeneralRatiosCalculator
from src.transformers.components.player_stats.int_converter import IntConverter
from src.transformers.components.player_stats.player_stats_general_merger import PlayerStatsGeneralMerger

class GeneralPlayerStatsPipeline(Pipeline):
    def __init__(self, raw_general_player_stats: pd.DataFrame):
        self.raw_general_player_stats = raw_general_player_stats
        super().__init__(steps=[
            ("raw_general_player_stats_merger", PlayerStatsGeneralMerger(raw_general_player_stats)),
            ("int_converter", IntConverter()),
            ("general_ratios_calculator", GeneralRatiosCalculator()),
            ("float_converter", FloatConverter())
        ])