import pandas as pd
from sklearn.pipeline import Pipeline

from src.transformers.player_stats.int_converter import IntConverter
from src.transformers.player_stats.player_stats_street_merger import PlayerStatsStreetMerger


class PostflopPlayerStatsPipeline(Pipeline):
    def __init__(
            self,
            flop_stats: pd.DataFrame,
            turn_stats: pd.DataFrame,
            river_stats: pd.DataFrame
    ):
        self.flop_stats = flop_stats
        self.turn_stats = turn_stats
        self.river_stats = river_stats
        super().__init__(steps=[
            ("flop_stats_merger", PlayerStatsStreetMerger(street_stats=flop_stats, street_name="flop")),
            ("turn_stats_merger", PlayerStatsStreetMerger(street_stats=turn_stats, street_name="turn")),
            ("river_stats_merger", PlayerStatsStreetMerger(street_stats=river_stats, street_name="river")),
            ("int_converter", IntConverter()),
            # ("postflop_ratios_calculator", PostflopRatiosCalculator())
        ])