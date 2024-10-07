import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_stats.int_converter import IntConverter
from src.transformers.player_stats.player_stats_street_merger import PlayerStatsStreetMerger

class PreflopPlayerStatsPipeline(Pipeline):
    def __init__(self, preflop_stats: pd.DataFrame):
        self.preflop_stats = preflop_stats
        super().__init__(steps=[
            ("preflop_stats_merger", PlayerStatsStreetMerger(street_stats=preflop_stats, street_name="preflop")),
            ("int_converter", IntConverter())
        ])
