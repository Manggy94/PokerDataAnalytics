import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_hand_stats.player_hand_stats_street_merger import HandStatsStreetMerger


class PostflopPlayerHandStatsPipeline(Pipeline):
    def __init__(self, streets: pd.DataFrame):
        self.streets = streets
        super().__init__(steps=[
            ("hand_stats_street_merger", HandStatsStreetMerger(streets)),
        ])