import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_stats.general_ratios_calculator import GeneralRatiosCalculator
from src.transformers.player_stats.int_converter import IntConverter
from src.transformers.player_stats.player_stats_general_merger import PlayerStatsGeneralMerger
from src.transformers.player_stats.player_stats_street_merger import PlayerStatsStreetMerger
from src.transformers.player_stats.preflop_ratios_calculator import PreflopRatiosCalculator

class PlayerStatsPipeline(Pipeline):

    def __init__(
            self,
            general_stats: pd.DataFrame,
            preflop_stats: pd.DataFrame,
    ):
        self.general_stats = general_stats
        self.preflop_stats = preflop_stats
        super().__init__(steps=[
            ("general_stats_merger", PlayerStatsGeneralMerger(player_general_stats=general_stats)),
            ("general_ratios_calculator", GeneralRatiosCalculator()),
            ("preflop_stats_merger", PlayerStatsStreetMerger(street_stats=preflop_stats, street_name="preflop")),
            ("preflop_ratios_calculator", PreflopRatiosCalculator()),
            ("int_converter", IntConverter())
        ])