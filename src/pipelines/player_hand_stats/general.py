import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_hand_stats.general.combos_merger import CombosMerger
from src.transformers.player_hand_stats.general.na_dropper import NaDropper
from src.transformers.player_hand_stats.general.positions_merger import PositionsMerger


class GeneralPlayerHandStatsPipeline(Pipeline):

    def __init__(self, positions: pd.DataFrame, combos: pd.DataFrame, streets: pd.DataFrame, action_moves: pd.DataFrame):
        self.action_moves = action_moves
        self.combos = combos
        self.positions = positions
        self.streets = streets
        super().__init__(steps=[
            ("na_dropper", NaDropper()),
            ("combos_merger", CombosMerger(combos)),
            ("positions_merger", PositionsMerger(positions)),
        ])