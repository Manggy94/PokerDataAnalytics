import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.hand_histories.hand_date_type_corrector import HandDateTypeCorrector
from src.transformers.hand_histories.hand_histories_combos_merger import HandHistoriesCombosMerger
from src.transformers.hand_histories.hand_histories_flops_merger import HandHistoriesFlopsMerger
from src.transformers.hand_histories.hand_histories_levels_merger import HandHistoriesLevelsMerger
from src.transformers.hand_histories.hand_histories_river_merger import HandHistoriesRiverMerger
from src.transformers.hand_histories.hand_histories_turn_merger import HandHistoriesTurnMerger
from src.transformers.hand_histories.hand_id_dropper import HandIdDropper
from src.transformers.boolean_converter import BooleanConverter
from src.transformers.float_converter import FloatConverter
from src.transformers.int_converter import IntConverter
from src.transformers.objects_categorizer import ObjectsCategorizer


class HandHistoriesPipeline(Pipeline):


    def __init__(
            self,
            cards: pd.DataFrame,
            combos: pd.DataFrame,
            flops: pd.DataFrame,
            levels: pd.DataFrame,
    ):
        self.cards = cards
        self.combos = combos
        self.flops = flops
        self.levels = levels
        super().__init__(steps=[
            ("hand_id_dropper", HandIdDropper()),
            ("hand_date_type_corrector", HandDateTypeCorrector()),
            ("hand_histories_levels_merger", HandHistoriesLevelsMerger(levels)),
            ("hand_histories_combos_merger", HandHistoriesCombosMerger(combos)),
            ("hand_histories_flops_merger", HandHistoriesFlopsMerger(flops)),
            ("hand_histories_turn_merger", HandHistoriesTurnMerger(cards)),
            ("hand_histories_river_merger", HandHistoriesRiverMerger(cards)),
            ("boolean_converter", BooleanConverter()),
            ("int_converter", IntConverter()),
            ("float_converter", FloatConverter()),
            ("objects_categorizer", ObjectsCategorizer())
        ])