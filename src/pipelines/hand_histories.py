import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.hand_histories.hand_date_type_corrector import HandDateTypeCorrector
from src.transformers.hand_histories.hand_histories_levels_merger import HandHistoriesLevelsMerger


class HandHistoriesPipeline(Pipeline):


    def __init__(self, levels: pd.DataFrame):
        self.levels = levels
        super().__init__(steps=[
            ("hand_date_type_corrector", HandDateTypeCorrector()),
            ("hand_histories_levels_merger", HandHistoriesLevelsMerger(levels))
        ])