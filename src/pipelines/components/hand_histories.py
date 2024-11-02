import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.components.hand_histories.hand_date_type_corrector import HandDateTypeCorrector
from src.transformers.components.hand_histories.hand_histories_combos_merger import HandHistoriesCombosMerger
from src.transformers.components.hand_histories.hand_histories_flops_merger import HandHistoriesFlopsMerger
from src.transformers.components.hand_histories.hand_histories_levels_merger import HandHistoriesLevelsMerger
from src.transformers.components.hand_histories.hand_histories_river_merger import HandHistoriesRiverMerger
from src.transformers.components.hand_histories.hand_histories_turn_merger import HandHistoriesTurnMerger
from src.transformers.components.hand_histories.hand_id_dropper import HandIdDropper
from src.transformers.components.hand_histories.hand_histories_tournaments_merger import HandHistoriesTournamentsMerger
from src.transformers.utils.boolean_converter import BooleanConverter
from src.transformers.utils.float_converter import FloatConverter
from src.transformers.utils.int_converter import IntConverter
from src.transformers.utils.na_bool_filler import NaBoolFiller
from src.transformers.utils.na_replacer import NaReplacer


class HandHistoriesPipeline(Pipeline):


    def __init__(
            self,
            cards: pd.DataFrame,
            combos: pd.DataFrame,
            flops: pd.DataFrame,
            levels: pd.DataFrame,
            tournaments: pd.DataFrame
    ):
        self.cards = cards
        self.combos = combos
        self.flops = flops
        self.levels = levels
        self.tournaments = tournaments
        super().__init__(steps=[
            ("hand_id_dropper", HandIdDropper()),
            ("hand_date_type_corrector", HandDateTypeCorrector()),
            ("hand_histories_tournaments_merger", HandHistoriesTournamentsMerger(tournaments)),
            ("hand_histories_levels_merger", HandHistoriesLevelsMerger(levels)),
            ("hand_histories_combos_merger", HandHistoriesCombosMerger(combos)),
            ("hand_histories_flops_merger", HandHistoriesFlopsMerger(flops)),
            ("hand_histories_turn_merger", HandHistoriesTurnMerger(cards)),
            ("hand_histories_river_merger", HandHistoriesRiverMerger(cards)),
            ("boolean_converter", BooleanConverter()),
            ("int_converter", IntConverter()),
            ("na_replacer_flop", NaReplacer(source_column="flop", keywords=["flop_is", "flop_has"])),
            ("na_replacer_flop1", NaReplacer(source_column="flop_first_card", keywords=["flop_first_card_"])),
            ("na_replacer_flop2", NaReplacer(source_column="flop_second_card", keywords=["flop_second_card_"])),
            ("na_replacer_flop3", NaReplacer(source_column="flop_third_card", keywords=["flop_third_card_"])),
            ("na_replacer_turn", NaReplacer(source_column="turn_card", keywords=["turn_card_"])),
            ("na_replacer_river", NaReplacer(source_column="river_card", keywords=["river_card_"])),
            ("na_bool_filler", NaBoolFiller()),
            ("float_converter", FloatConverter()),
            # ("objects_categorizer", ObjectsCategorizer())
        ])