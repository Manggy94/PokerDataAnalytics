import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_stats.int_converter import IntConverter
from src.transformers.player_stats.player_stats_street_merger import PlayerStatsStreetMerger
from src.transformers.player_stats.preflop_ratios_calculator import PreflopRatiosCalculator


class PreflopPlayerStatsPipeline(Pipeline):
    def __init__(self, raw_preflop_player_stats: pd.DataFrame):
        self.raw_preflop_player_stats = raw_preflop_player_stats
        super().__init__(steps=[
            ("raw_preflop_player_stats_merger", PlayerStatsStreetMerger(street_stats=raw_preflop_player_stats, street_name="preflop")),
            ("int_converter", IntConverter()),
            ("preflop_ratios_calculator", PreflopRatiosCalculator())
        ])
