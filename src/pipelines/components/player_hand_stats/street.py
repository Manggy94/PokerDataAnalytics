import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.components.player_hand_stats.player_hand_stats_move_merger import HandStatsMoveMerger
from src.transformers.components.player_hand_stats.player_hand_stats_sequence_merger import HandStatsSequenceMerger
from src.transformers.utils.boolean_converter import BooleanConverter
from src.transformers.utils.float_converter import FloatConverter
from src.transformers.utils.int_converter import IntConverter
from src.transformers.utils.na_bool_filler import NaBoolFiller


class StreetPlayerHandStatsPipeline(Pipeline):
    def __init__(self, action_moves: pd.DataFrame, sequences: pd.DataFrame):
        self.action_moves = action_moves
        self.sequences = sequences
        super().__init__(steps=[
            ("hand_stats_move_merger", HandStatsMoveMerger(action_moves)),
            ("hand_stats_sequence_merger", HandStatsSequenceMerger(sequences)),
            ("boolean_converter", BooleanConverter()),
            ("na_bool_filler", NaBoolFiller()),
            ("int_converter", IntConverter()),
            ("float_converter", FloatConverter()),
            # ("objects_categorizer", ObjectsCategorizer())
        ])