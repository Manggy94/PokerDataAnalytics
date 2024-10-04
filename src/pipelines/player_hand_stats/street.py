import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_hand_stats.player_hand_stats_move_merger import HandStatsMoveMerger
from src.transformers.player_hand_stats.player_hand_stats_sequence_merger import HandStatsSequenceMerger
from src.transformers.boolean_converter import BooleanConverter
from src.transformers.float_converter import FloatConverter
from src.transformers.int_converter import IntConverter
from src.transformers.objects_categorizer import ObjectsCategorizer


class StreetPlayerHandStatsPipeline(Pipeline):
    def __init__(self, action_moves: pd.DataFrame, sequences: pd.DataFrame):
        self.action_moves = action_moves
        self.sequences = sequences
        super().__init__(steps=[
            ("hand_stats_move_merger", HandStatsMoveMerger(action_moves)),
            ("hand_stats_sequence_merger", HandStatsSequenceMerger(sequences)),
            ("boolean_converter", BooleanConverter()),
            ("int_converter", IntConverter()),
            ("float_converter", FloatConverter()),
            ("objects_categorizer", ObjectsCategorizer())
        ])