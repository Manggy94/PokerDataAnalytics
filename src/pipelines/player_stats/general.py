import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_stats.general_ratios_calculator import GeneralRatiosCalculator
from src.transformers.player_stats.int_converter import IntConverter
from src.transformers.player_stats.player_stats_general_merger import PlayerStatsGeneralMerger
from src.transformers.float_converter import FloatConverter

class GeneralPlayerStatsPipeline(Pipeline):
    def __init__(self, general_stats: pd.DataFrame):
        self.general_stats = general_stats
        super().__init__(steps=[
            ("general_stats_merger", PlayerStatsGeneralMerger(general_stats)),
            ("int_converter", IntConverter()),
            ("general_ratios_calculator", GeneralRatiosCalculator()),
            ("float_converter", FloatConverter())
        ])