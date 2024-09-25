import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.hand_histories.hand_date_type_corrector import HandDateTypeCorrector
from src.transformers.hand_histories.hand_histories_combos_merger import HandHistoriesCombosMerger
from src.transformers.hand_histories.hand_histories_flops_merger import HandHistoriesFlopsMerger
from src.transformers.hand_histories.hand_histories_levels_merger import HandHistoriesLevelsMerger


class HandHistoriesPipeline(Pipeline):


    def __init__(self, levels: pd.DataFrame, flops: pd.DataFrame, cards:pd.DataFrame, combos: pd.DataFrame):
        self.levels = levels
        self.flops = flops
        self.combos = combos
        self.cards = cards
        super().__init__(steps=[
            ("hand_date_type_corrector", HandDateTypeCorrector()),
            ("hand_histories_levels_merger", HandHistoriesLevelsMerger(levels)),
            ("hand_histories_flops_merger", HandHistoriesFlopsMerger(flops)),
            ("hand_histories_combos_merger", HandHistoriesCombosMerger(combos))
        ])