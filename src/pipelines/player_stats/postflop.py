import pandas as pd
from sklearn.pipeline import Pipeline

from src.transformers.player_stats.int_converter import IntConverter
from src.transformers.player_stats.player_stats_street_merger import PlayerStatsStreetMerger
from src.transformers.player_stats.postflop_ratios_calculator import PostflopRatiosCalculator


class PostflopPlayerStatsPipeline(Pipeline):
    def __init__(
            self,
            raw_flop_player_stats: pd.DataFrame,
            raw_turn_player_stats: pd.DataFrame,
            raw_river_player_stats: pd.DataFrame
    ):
        self.raw_flop_player_stats = raw_flop_player_stats
        self.raw_turn_player_stats = raw_turn_player_stats
        self.raw_river_player_stats = raw_river_player_stats
        super().__init__(steps=[
            ("flop_stats_merger", PlayerStatsStreetMerger(street_stats=raw_flop_player_stats, street_name="flop")),
            ("turn_stats_merger", PlayerStatsStreetMerger(street_stats=raw_turn_player_stats, street_name="turn")),
            ("river_stats_merger", PlayerStatsStreetMerger(street_stats=raw_river_player_stats, street_name="river")),
            ("int_converter", IntConverter()),
            ("flop_ratios_calculator", PostflopRatiosCalculator("flop")),
            ("turn_ratios_calculator", PostflopRatiosCalculator("turn")),
            ("river_ratios_calculator", PostflopRatiosCalculator("river"))
        ])