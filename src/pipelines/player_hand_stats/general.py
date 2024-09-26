import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_hand_stats.general.positions_merger import PositionsMerger


class GeneralPlayerHandStatsPipeline(Pipeline):

    def __init__(self, positions: pd.DataFrame, combos: pd.DataFrame):
        self.combos = combos
        self.positions = positions
        super().__init__(steps=[
            ("positions_merger", PositionsMerger(positions))
        ])