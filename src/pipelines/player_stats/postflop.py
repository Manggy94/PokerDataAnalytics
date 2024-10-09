import pandas as pd
from sklearn.pipeline import Pipeline

from src.transformers.int_converter import IntConverter
from src.transformers.player_stats.player_stats_street_merger import PlayerStatsStreetMerger


class PostflopPlayerStatsPipeline(Pipeline):
    def __init__(self, postflop_stats: pd.DataFrame, street_name: str):
        self.postflop_stats = postflop_stats
        self.street_name = street_name
        super().__init__(steps=[
            ("postflop_stats_merger", PlayerStatsStreetMerger(street_stats=postflop_stats, street_name=street_name)),
            ("int_converter", IntConverter()),
            # ("postflop_ratios_calculator", PostflopRatiosCalculator())
        ])