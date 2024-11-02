import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.components.player_hand_stats.general.combos_merger import CombosMerger
from src.transformers.components.player_hand_stats.general.na_dropper import NaDropper
from src.transformers.components.player_hand_stats.general.positions_merger import PositionsMerger
from src.transformers.components.player_hand_stats.general.raw_dtypes_corrector import RawDtypesCorrector
from src.transformers.components.player_hand_stats.general.seats_categorizer import SeatsCategorizer
from src.transformers.components.player_hand_stats.player_hand_stats_move_merger import HandStatsMoveMerger
from src.transformers.components.player_hand_stats.player_hand_stats_action_street_merger import HandStatsActionStreetMerger
from src.transformers.utils.boolean_converter import BooleanConverter
from src.transformers.utils.float_converter import FloatConverter
from src.transformers.utils.int_converter import IntConverter
from src.transformers.utils.na_bool_filler import NaBoolFiller
from src.transformers.utils.na_replacer import NaReplacer


class GeneralPlayerHandStatsPipeline(Pipeline):

    def __init__(
            self,
            action_moves: pd.DataFrame,
            combos: pd.DataFrame,
            positions: pd.DataFrame,
            streets: pd.DataFrame,

    ):
        self.action_moves = action_moves
        self.combos = combos
        self.positions = positions
        self.streets = streets
        super().__init__(steps=[
            ('"raw_dtypes_corrector', RawDtypesCorrector()),
            ("na_dropper", NaDropper()),
            ("seats_categorizer", SeatsCategorizer()),
            ("combos_merger", CombosMerger(combos)),
            ("positions_merger", PositionsMerger(positions)),
            ("hand_stats_street_merger", HandStatsActionStreetMerger(streets)),
            ("hand_stats_move_merger", HandStatsMoveMerger(action_moves)),
            ("boolean_converter", BooleanConverter()),
            ("na_replacer", NaReplacer(source_column="player_combo", keywords=["player_combo_"])),
            ("na_bool_filler", NaBoolFiller()),
            ("int_converter", IntConverter()),
            ("float_converter", FloatConverter()),
        ])