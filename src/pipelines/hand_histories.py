import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.hand_histories.hand_date_type_corrector import HandDateTypeCorrector



class HandHistoriesPipeline(Pipeline):


    def __init__(self):
        super().__init__(steps=[
            ("hand_date_type_corrector", HandDateTypeCorrector())
        ])