import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_hand_stats.player_hand_stats_move_merger import HandStatsMoveMerger


class PreflopPlayerHandStatsPipeline(Pipeline):
    def __init__(self, action_moves: pd.DataFrame):
        self.action_moves = action_moves
        super().__init__(steps=[
            ("hand_stats_move_merger", HandStatsMoveMerger(action_moves))
        ])