from sklearn.pipeline import Pipeline
from src.transformers.positions.columns_cleaner import PositionsColumnsCleaner


class PositionsPipeline(Pipeline):
    def __init__(self):
        super().__init__(steps=[
            ("positions_cleaner", PositionsColumnsCleaner())
        ])